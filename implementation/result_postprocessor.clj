#!/usr/bin/env bb

(require '[clojure.string :as str]
         '[cheshire.core :as json]
         '[clojure.java.shell :refer [sh]])

(def default-radii {"balanced_triangle_classification_dataset" 2
                    "NCI1" 8
                    "PROTEINS_full" 5
                    "DD" 2
                    "REDDIT-BINARY" 1
                    "IMDB-BINARY" 8})
(def single-depth-radii (assoc default-radii
                               "IMDB-BINARY" 4))
(def default-iterations {"WL_st" 5
                         "WL_sp" 5})
(def ds-rename {"balanced_triangle_classification_dataset" "TRIANGLE"
                "PROTEINS_full" "PROTEINS"
                "DD" "D\\&D"
                "REDDIT-BINARY" "REDDIT"
                "IMDB-BINARY" "IMDB"})
(def ds-colnames {"balanced_triangle_classification_dataset" "triangle"
                  "NCI1" "nci"
                  "PROTEINS_full" "proteins"
                  "PROTEINS" "proteins"
                  "DD" "dd"
                  "REDDIT-BINARY" "reddit"
                  "IMDB-BINARY" "imdb"})
(def datasets ["balanced_triangle_classification_dataset"
               "NCI1" "PROTEINS_full" "DD" "REDDIT-BINARY" "IMDB-BINARY"])
(def models-with-potential-oom #{"WL\\textsubscript{SP}"})

(defn round
  ([num] (round num 0))
  ([num prec]
   (if (int? num)
     num
     (if (zero? prec)
       (int (Math/round num))
       (let [p (Math/pow 10 prec)]
         (/ (Math/round (* num p)) p))))))

(defn dim-str
  [feat lab]
  (if (zero? (+ feat lab))
    "0 + 1"
    (str feat " + " lab)))

(defn stats-dict->csv-line
  [[name {:strs [node_counts
                 edge_counts
                 dim_node_features
                 dim_edge_features
                 num_node_labels
                 num_edge_labels
                 node_degrees]}]]
  (str/join ","
            [(ds-rename name name)
             (get node_counts "count") ; graph count
             (round (get node_counts "min"))
             (round (get node_counts "mean"))
             (round (get node_counts "max"))
             (round (get edge_counts "min"))
             (round (get edge_counts "mean"))
             (round (get edge_counts "max"))
             (dim-str dim_node_features num_node_labels)
             (dim-str dim_edge_features num_edge_labels)
             (round (get node_degrees "min"))
             (round (get node_degrees "mean"))
             (round (get node_degrees "max"))]))

(defn ds-stats->csv
  []
  (let [stats (json/parse-string (slurp "data/stats.json"))
        head (str "name,"
                  "graph_count,"
                  "node_count_min,node_count_mean,node_count_max,"
                  "edge_count_min,edge_count_mean,edge_count_max,"
                  "dim_node_features,dim_edge_features,",
                  "node_degree_min,node_degree_mean,node_degree_max")
        stats (str head "\n" (str/join "\n" (map stats-dict->csv-line stats)) "\n")]
    (spit "../thesis/data/ds_stats.csv" stats)
    (println stats)))

(defn extract-pool
  [name]
  (let [pool (if (str/includes? name "Avg") "\\mean" "\\mathrm{SAM}")]
    pool
    #_(if (str/ends-with? name "FC")
        (str pool " + \\mathrm{MLP}")
        pool)))

(defn eval-name->params
  [dataset name]
  (let [[_ r n _ T] (re-find #"n?(\d?)_?(.+?)(_FC)?(_\d)?$" name)
        ;_ (println "y" name dataset r n T)
        r (if (seq r) (Integer/parseInt r) (default-radii dataset))
        T (when (seq T) (Integer/parseInt (subs T 1)))
        pool (extract-pool name)]
    (condp #(str/includes? %2 %1) n
      "CWL2GCN"
      {:name "2-WL-GNN"
       :order [1 4 (if (str/includes? pool "MLP") 0 1) r]
       :radius (str "r=" r)
       :is-default (= r (single-depth-radii dataset))
       :is-lta (not (str/ends-with? name "FC"))
       :pool pool}
      "K2GNN"
      {:name "2-GNN"
       :order [1 3 0 0]
       :is-default true
       :pool pool}
      "WL_st"
      {:name "WL\\textsubscript{ST}"
       :order [0 1 (or T 5) 0]
       :it (str "T=" (or T 5))
       :is-lta true
       :is-default (or (= T 1) (nil? T))}
      "WL_sp"
      {:name "WL\\textsubscript{SP}"
       :order [0 2 (or T 5) 0]
       :it (str "T=" (or T 5))
       :is-default (nil? T)}
      "LWL2"
      {:name "2-LWL"
       :order [0 3 (or T 3) 0]
       :it (str "T=" (or T 3))
       :is-lta true
       :is-default (nil? T)}
      "GWL2"
      {:name "2-GWL"
       :order [0 4 (or T 3) 0]
       :it (str "T=" (or T 3))
       :is-default (nil? T)}
      "GIN"
      {:name "GIN" :order [1 2 0 0]
       :pool "\\mathrm{sum}"
       :is-default true}
      "MolecularFingerprint"
      {:name "Baseline" :order [1 1 0 0]
       :pool "\\mathrm{sum}"
       :is-default true}
      "DeepMultisets"
      {:name "Baseline" :order [1 1 0 0]
       :pool "\\mathrm{sum}"
       :is-default true}
      nil)))

(defn to-proc
  [x]
  (* 100 (or x 0)))

(defn dataset-results
  [dataset & {:keys [only-default] :or {only-default true}}]
  (let [{evals :out} (sh "ls" "./evaluations/")
        evals (str/split evals #"\n+")
        summaries (into []
                        (comp (filter #(and (str/starts-with? % dataset) (not (str/ends-with? % "quick"))))
                              (map (juxt identity #(try (slurp (str "./evaluations/" % "/summary/results.json")) (catch Exception _))))
                              (keep (fn [[name s]] (when s (assoc (json/parse-string s true)
                                                                  :name (subs name (inc (count dataset)))))))
                              (keep (fn [{name :name
                                          folds :folds
                                          {test :accuracy} :combined_test
                                          {train :accuracy} :combined_train}]
                                      (when (= (:count test) 10) ; only include evaluations of all 10-folds
                                        {:name name
                                         :test-mean (to-proc (:mean test))
                                         :test-std (to-proc (:std test))
                                         :train-mean (to-proc (:mean train))
                                         :train-std (to-proc (:std train))
                                         :folds (map (fn [{{test :accuracy} :test}]
                                                       {:test-mean (to-proc (:mean test))})
                                                     folds)}))))
                        evals)
        {comp-evals :out} (sh "ls" "./libs/gnn-comparison/RESULTS/")
        comp-evals (str/split comp-evals #"\n+")
        summaries (into summaries
                        (comp (filter #(str/ends-with? % (str (if (= dataset "PROTEINS_full") "PROTEINS" dataset) "_assessment")))
                              (map (juxt identity
                                         (fn [dir]
                                           (try
                                             {:folds (mapv #(slurp (str "./libs/gnn-comparison/RESULTS/" dir "/10_NESTED_CV/"
                                                                        "OUTER_FOLD_" % "/outer_results.json"))
                                                          (range 1 11))
                                              :res (slurp (str "./libs/gnn-comparison/RESULTS/" dir "/10_NESTED_CV/assessment_results.json"))}
                                             (catch Exception e (println e))))))
                              (keep (fn [[name {:keys [folds res] :or {folds nil res nil}}]]
                                      (when res
                                        (assoc (json/parse-string res true)
                                               :name name
                                               :folds (map #(json/parse-string % true) folds)))))
                              (keep (fn [{:keys [name folds avg_TR_score std_TR_score avg_TS_score std_TS_score]}]
                                      {:name name
                                       :test-mean avg_TS_score
                                       :test-std std_TS_score
                                       :train-mean avg_TR_score
                                       :train-std std_TR_score
                                       :folds (map (fn [{:keys [OUTER_TS]}] {:test-mean OUTER_TS})
                                                   folds)})))
                        comp-evals)
        results (keep (fn [{name :name :as sum}]
                        (when-let [params (eval-name->params dataset name)]
                          (when (or (not only-default) (:is-default params))
                            (merge sum {:model (params :name)
                                        :order (params :order)
                                        :dataset dataset
                                        :is-default (:is-default params)
                                        :is-lta (:is-lta params)
                                        :pool (or (:pool params) "")
                                        :it (or (:it params) "")
                                        :params (str/join ", " (keep params [:pool :it]))}))))
                      summaries)
        typed-max (fn ([] {}) ([x] x) ([max-dict [t v]] (update max-dict t (fnil max 0) v)))
        max-grouper (juxt (comp first :order) :is-default)
        max-test (transduce (map (juxt max-grouper :test-mean)) typed-max {} results)
        max-train (transduce (map (juxt max-grouper :train-mean)) typed-max {} results)
        results (map (fn [res] (assoc res
                                      :is-best-test (= (max-test (max-grouper res)) (:test-mean res))
                                      :is-best-train (= (max-train (max-grouper res)) (:train-mean res))))
                     results)]
      results))

(defn dataset-result-head
  [dataset]
  (let [pre (ds-colnames dataset)]
    (str pre "TestMean;" pre "TestStd;" pre "TrainMean;" pre "TrainStd;" pre "BestTest;" pre "BestTrain")))

(defn dataset-result-row
  [datasets results idx [model params is-default is-lta]]
  (let [results (into {}
                      (comp (filter #(and (= (:model %) model)
                                          (= (:params %) params)
                                          (= (:is-default %) is-default)
                                          (= (:is-lta %) is-lta)))
                            (map (juxt :dataset identity)))
                      results)]
    (str idx ";" model ";" (when (seq params) (str params)) ";"
         (if is-default "1" "0") ";" (if is-lta "1" "0") ";"
         (str/join ";" (map (fn [ds]
                              (if-let [res (results ds)]
                                (str (str/join ";" (->> [:test-mean :test-std :train-mean :train-std]
                                                        (map res)
                                                        (map #(format "%.1f" (double %)))))
                                     ";" (if (:is-best-test res) "1" "0")
                                     ";" (if (:is-best-train res) "1" "0"))
                                (if (models-with-potential-oom model) "m;m;m;m;0;0" "t;t;t;t;0;0")))
                            datasets)))))

(defn eval-results->csv
  [{:keys [only-default]} file & args]
  (let [datasets (if (empty? args) datasets args)
        results (sort-by :order (mapcat #(dataset-results % :only-default only-default) datasets))
        models-with-params (distinct (map (juxt :model :params :is-default :is-lta) results))
        head (str "id;model;params;isDefault;isLta;" (str/join ";" (map dataset-result-head datasets)))
        rows (into [] (map-indexed (partial dataset-result-row datasets results)) models-with-params)
        csv (str head "\n" (str/join "\n" rows) "\n")]
    (spit (str "../thesis/data/" file ".csv") csv)
    (println csv)))

(defn res->full-name
  [{name :model params :params is-lta :is-lta}]
  (str "\\textbf{"
       (if is-lta
         (str "\\textcolor{t_darkgreen}{" name "*}")
         name)
       "} ($" params "$)"))

(defn mean
  [vals]
  (/ (apply + vals) (count vals)))

(defn std
  [vals]
  (let [m (mean vals)]
    [m (Math/sqrt
         (/ (apply + (map (comp #(* % %) #(- % m))
                          vals))
            (count vals)))]))

(defn fold-differences->csv
  [{:keys [only-default]} file dataset]
  (let [results (sort-by :order (dataset-results dataset :only-default only-default))
        diffs
        (for [{folds_a :folds} results
              {folds_b :folds} results
              :let [test_a (map :test-mean folds_a)
                    test_b (map :test-mean folds_b)
                    test_diffs (map - test_a test_b)
                    [test_diff_mean test_diff_std] (std test_diffs)
                     d (format "$%.1f \\pm %.1f$"
                               (double test_diff_mean)
                               (double test_diff_std))]]
          (cond
            (pos? test_diff_mean)
            (str "\\textcolor{t_darkgreen}{" d "}")
            (neg? test_diff_mean)
            (str "\\textcolor{t_red}{" d "}")
            (zero? test_diff_std) ""
            :else d))
        diffs (partition (count results) diffs)
        head (str ";" (str/join ";" (map (comp #(str "\\rotatebox[origin=c]{90}{" % "}")
                                               res->full-name)
                                         results)))
        rows (map (fn [res diffs]
                    (str (res->full-name res) ";"
                         (str/join ";" diffs)))
                  results diffs)
        csv (str head "\n" (str/join "\n" rows) "\n")]
    (spit (str "../thesis/data/" file ".csv") csv)
    (println csv)))

(def actions {"ds_stats" ds-stats->csv
              "eval_res_full" (partial eval-results->csv {:only-default false})
              "eval_res" (partial eval-results->csv {:only-default true})
              "fold_diff" (partial fold-differences->csv {:only-default true})})

(println "LTAG Results Postprocessor.")
(if-let [action (actions (first *command-line-args*))]
  (apply action (rest *command-line-args*))
  (println "Unknown action:" (first *command-line-args*)))
(println "Done.")
