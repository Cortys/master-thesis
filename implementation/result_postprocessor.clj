#!/usr/bin/env bb

(require '[clojure.string :as str]
         '[cheshire.core :as json]
         '[clojure.java.shell :refer [sh]])

(def include-lta true)

(def default-radii {"balanced_triangle_classification_dataset" 2
                    "hyperloop_dataset" 1
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
(def wlst-model "WL\\textsubscript{ST}")
(def wlsp-model "WL\\textsubscript{SP}")
(def k2gnn-model "2-GNN")
(def wl2-gnn-model "2-WL-GNN")
(def mean-pool "\\mean")
(def sam-pool (if include-lta "\\mathrm{SAM}" "\\wmean"))
(def models-with-potential-oom #{wlst-model wlsp-model k2gnn-model})

(defn round
  ([num] (round num 0))
  ([num prec]
   (if (int? num)
     num
     (if (zero? prec)
       (int (Math/round num))
       (let [p (Math/pow 10 prec)
             rounded (/ (Math/round (* num p)) p)]
         (if (pos? prec)
           rounded
           (int rounded)))))))

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
                 node_degrees
                 radii]}]]
  (str/join ","
            [(ds-rename name name)
             (get node_counts "count") ; graph count
             (round (get node_counts "min"))
             (round (get node_counts "mean") 1)
             (round (get node_counts "max"))
             (round (get edge_counts "min"))
             (round (get edge_counts "mean") 1)
             (round (get edge_counts "max"))
             (dim-str dim_node_features num_node_labels)
             (dim-str dim_edge_features num_edge_labels)
             (round (get node_degrees "min"))
             (round (get node_degrees "mean") 1)
             (round (get node_degrees "max"))
             (round (get radii "mean") 1)
             (round (get radii "std") 1)]))

(defn ds-stats->csv
  []
  (let [stats (json/parse-string (slurp "data/stats.json"))
        head (str "name,"
                  "graph_count,"
                  "node_count_min,node_count_mean,node_count_max,"
                  "edge_count_min,edge_count_mean,edge_count_max,"
                  "dim_node_features,dim_edge_features,",
                  "node_degree_min,node_degree_mean,node_degree_max,"
                  "radius_mean,radius_std")
        stats (str head "\n" (str/join "\n" (map stats-dict->csv-line stats)) "\n")]
    (spit "../thesis/data/ds_stats.csv" stats)
    (println stats)))

(defn extract-pool
  [name]
  (let [pool (if (str/includes? name "Avg") mean-pool sam-pool)]
    pool
    #_(if (str/ends-with? name "FC")
        (str pool " + \\mathrm{MLP}")
        pool)))

(defn eval-name->params
  [dataset name]
  (let [[_ r n _ T time-eval] (re-find #"n?(\d?)_?(.+?)(_FC)?(_\d)?(_time_eval)?$" name)
        ;_ (println "y" name dataset r n T)
        r (if (seq r) (Integer/parseInt r) (default-radii dataset))
        T (when (seq T) (Integer/parseInt (subs T 1)))
        pool (extract-pool name)
        time-eval (some? time-eval)]
    (condp #(str/includes? %2 %1) n
      "CWL2GCN"
      {:name "2-WL-GNN"
       :order [1 4 (if (str/includes? pool "MLP") 0 1) r]
       :r r
       :is-default (= r (single-depth-radii dataset))
       :is-lta (not (str/ends-with? name "FC"))
       :time-eval time-eval
       :pool pool}
      "K2GNN"
      {:name "2-GNN"
       :order [1 3 0 0]
       :is-default (= r (default-radii dataset))
       :time-eval time-eval
       :pool pool}
      "WL_st"
      {:name wlst-model
       :order [0 1 (or T 5) 0]
       :it (str "T=" (or T 5))
       :T (or T 5)
       :is-lta true
       :is-default (or (= T 1) (= T 3))
       :hide-diff (= T 1)}
      "WL_sp"
      {:name wlsp-model
       :order [0 2 (or T 5) 0]
       :it (str "T=" (or T 5))
       :is-default (= T 3)}
      "LWL2"
      {:name "2-LWL"
       :order [0 3 (or T 3) 0]
       :it (str "T=" (or T 3))
       :T (or T 3)
       :is-lta true
       :is-default (nil? T)}
      "GWL2"
      {:name "2-GWL"
       :order [0 4 (or T 3) 0]
       :it (str "T=" (or T 3))
       :T (or T 3)
       :is-default (nil? T)}
      "GIN"
      {:name "GIN" :order [1 2 0 0]
       :pool "\\mathrm{sum}"
       :time-eval time-eval
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

(defn ls-dir
  [dir]
  (let [{out :out} (sh "ls" dir)]
    (str/split out #"\n+")))

(defn dataset-results
  [dataset & {:keys [only-default] :or {only-default true}}]
  (let [evals (ls-dir "./evaluations/")
        summaries (into []
                        (comp (filter #(and (str/starts-with? % dataset) (not (str/ends-with? % "quick"))))
                              (map (juxt identity #(try
                                                     [(slurp (str "./evaluations/" % "/summary/results.json"))
                                                      (slurp (str "./evaluations/" % "/config.json"))]
                                                     (catch Exception _))))
                              (keep (fn [[name [s c]]]
                                      (when (and c s)
                                        (assoc (json/parse-string s true)
                                               :name (subs name (inc (count dataset)))
                                               :config (json/parse-string c true)))))
                              (keep (fn [{name :name
                                          config :config
                                          folds :folds
                                          {test :accuracy} :combined_test
                                          {train :accuracy} :combined_train}]
                                      (when (= (:count test) 10) ; only include evaluations of all 10-folds
                                        {:name name
                                         :config config
                                         :test-mean (to-proc (:mean test))
                                         :test-std (to-proc (:std test))
                                         :train-mean (to-proc (:mean train))
                                         :train-std (to-proc (:std train))
                                         :folds (map (fn [{{test :accuracy} :test}]
                                                       {:test-mean (to-proc (:mean test))})
                                                     folds)}))))
                        evals)
        comp-evals (ls-dir "./libs/gnn-comparison/RESULTS/")
        summaries (into summaries
                        (comp (filter #(str/ends-with? % (str (if (= dataset "PROTEINS_full") "PROTEINS" dataset) "_assessment")))
                              (map (juxt identity
                                         (fn [dir]
                                           (try
                                             {:folds (mapv #(try
                                                              (slurp (str "./libs/gnn-comparison/RESULTS/" dir "/10_NESTED_CV/"
                                                                          "OUTER_FOLD_" % "/outer_results.json"))
                                                              (catch Exception _))
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
                                        :hide-diff (:hide-diff params)
                                        :is-lta (:is-lta params)
                                        :pool (or (:pool params) "")
                                        :it (or (:it params) "")
                                        :T (:T params)
                                        :r (:r params)
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
  [dataset & {with-best :with-best :or {with-best true}}]
  (let [pre (ds-colnames dataset)]
    (if with-best
      (str pre "TestMean;" pre "TestStd;" pre "TrainMean;" pre "TrainStd;" pre "BestTest;" pre "BestTrain")
      (str pre "TestMean;" pre "TestStd;" pre "TrainMean;" pre "TrainStd"))))

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
       (if include-lta
         (cond
           is-lta (str "\\textcolor{t_darkgreen}{" name "*}")
           (= name "2-WL-GNN") (str name "\\phantom{*}")
           :else name)
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

(defn fold-compare
  [folds-a folds-b]
  (let [test-a (map :test-mean folds-a)
        test-b (map :test-mean folds-b)
        test_diffs (map - test-a test-b)]
    (std test_diffs)))

(defn fold-differences->tex
  [{:keys [only-default]} file dataset]
  (let [results (sort-by :order (dataset-results dataset :only-default only-default))
        results (if (> (count results) 12)
                  (remove :hide-diff results)
                  results)
        results (if include-lta results (remove #(and (:is-lta %) (= (:model %) "2-WL-GNN")) results))
        diffs
        (for [{folds-a :folds test-a-mean :test-mean test-a-std :test-std} results
              {folds-b :folds test-b-mean :test-mean test-b-std :test-std} results
              :let [[test-diff-mean test-diff-std]
                    (if (and (not-any? (comp nil? :test-mean) folds-a)
                             (not-any? (comp nil? :test-mean) folds-b))
                      (fold-compare folds-a folds-b)
                      [(- test-a-mean test-b-mean)
                       (if (not= test-a-std test-b-std)
                         (+ test-a-std test-b-std) ; assuming some fold-based dependence if data missing
                         0)])
                    test-diff-mean-str (cond
                                         (>= test-diff-mean 10) (str "+" (round test-diff-mean 0))
                                         (<= test-diff-mean -10) (str "-" (- (round test-diff-mean 0)))
                                         :else (format "%+.1f" (double test-diff-mean)))
                    test-diff-std-str (if (>= test-diff-std 10)
                                        (str "\\pm " (round test-diff-std 0))
                                        (format "\\pm %.1f" (double test-diff-std)))
                    d (format "{\\tiny\\Vectorstack{%s\\\\ %s}}"
                              test-diff-mean-str test-diff-std-str)]]
          (cond
            (and (pos? test-diff-mean) (pos? (- test-diff-mean (* 2 test-diff-std))))
            (str "\\cellcolor{t_green!25}\\textcolor{t_darkgreen}{" d "}")
            (and (neg? test-diff-mean) (neg? (+ test-diff-mean (* 2 test-diff-std))))
            (str "\\cellcolor{t_red!25}\\textcolor{t_darkred}{" d "}")
            (and (zero? test-diff-mean) (zero? test-diff-std)) ""
            :else d))
        diffs (partition (count results) diffs)
        head (str "& " (str/join " & " (map (comp #(str "\\rotatebox[origin=l]{90}{" % "}")
                                                  res->full-name)
                                            results)))
        rows (map (fn [res diffs]
                    (str (res->full-name res) "&"
                         (str/join " & " diffs)))
                  results diffs)
        tex (str "% This file was generated by the LTAG results postprocessor. Do not edit manually.\n"
                 "{\\setlength\\tabcolsep{2.5pt}\\setlength{\\extrarowheight}{2pt}%\n"
                 "\\begin{tabular}{l" (str/join "" (repeat (count rows) "c")) "}\n"
                 head " \\\\\n" (str/join " \\\\[2pt]\n" rows) "\n"
                 "\\end{tabular}}\n")]
    (spit (str "../thesis/data/" file ".tex") tex)
    (println tex)))

(defn all-fold-differences->tex
  []
  (doseq [dataset datasets
          :let [fname (ds-colnames dataset)]]
    (fold-differences->tex {:only-default true}
                           (str "diffs/" fname) dataset)))

(defn wl-depth-compare
  [{:keys [use-gnn pool] :or {use-gnn false pool sam-pool}} file]
  (let [model-filter (if use-gnn
                       #(and (= wl2-gnn-model (:model %)) (= pool (:pool %)))
                       #(= wlst-model (:model %)))
        depth-key (if use-gnn :r :T)
        results (sort-by :order (mapcat #(dataset-results % :only-default false) datasets))
        results (group-by depth-key (filter model-filter results))
        Ts (sort (keys results))
        head (str (name depth-key) ";" (str/join ";" (map #(dataset-result-head % :with-best false) datasets)))
        rows (map (fn [T]
                    (let [T-results (results T)
                          vals (mapcat #((juxt :test-mean :test-std :train-mean :train-std)
                                         (or (first (filter (comp (partial = %) :dataset) T-results))
                                             {:test-mean "nan" :test-std "nan" :train-mean "nan" :train-std "nan"}))
                                       datasets)]
                      (str T ";" (str/join ";" vals))))
                  Ts)
        csv (str head "\n" (str/join "\n" rows) "\n")]
    (spit (str "../thesis/data/" file ".csv") csv)
    (println csv)))

(defn wl2-single-radius->csv
  [r res]
  (let [head (str "ds;"
                  "samTestMean;" "samTestStd;" "samTrainMean;" "samTrainStd;"
                  "meanTestMean;" "meanTestStd;" "meanTrainMean;" "meanTrainStd")
        datasets (distinct (map :dataset res))
        results (group-by :pool res)
        rows (for [ds datasets
                   :let [mean-res (first (filter #(= ds (:dataset %)) (results mean-pool)))
                         sam-res (first (filter #(= ds (:dataset %)) (results sam-pool)))
                         selector (juxt :test-mean :test-std :train-mean :train-std)]]
               (str (ds-rename ds ds) ";" (str/join ";" (concat (selector sam-res) (selector mean-res)))))
        csv (str head "\n" (str/join "\n" rows) "\n")]
    (println r csv)
    (spit (str "../thesis/data/wl2_radii/r_" r ".csv") csv)))

(defn wl2-radius-compare
  []
  (let [results (into []
                      (comp (remove #{"REDDIT-BINARY"})
                            (mapcat #(dataset-results % :only-default false))
                            (filter #(and (= (:model %) "2-WL-GNN") (:is-lta %))))
                      datasets)
        results (sort-by :order results)
        results (group-by :r results)
        radii (sort (keys results))]
    (doseq [r radii] (wl2-single-radius->csv r (results r)))))

(defn durations
  []
  (let [results (sort-by :order (mapcat #(dataset-results % :only-default true) datasets))
        results (->> results
                     (filter :config)
                     (remove :T)
                     (filter (comp #{"NCI1"}
                                   :dataset))
                     ;(filter #(= (:model %) "2-WL-GNN"))
                     (map #(update-in % [:config :duration] / (* 60 60 24))))
        durs (map (comp :duration :config) results)
        min-dur (apply min durs)
        max-dur (apply max durs)]
    (println "Min:" min-dur "Max:" max-dur "Mean/std:" (std durs))
    (doseq [res results]
      (println "Duration for " (:model res) "-" (:dataset res) "-" (:params res) "-" (:duration (:config res))))))

(defn time-eval-name->params
  [name]
  (let [[_ factor-str _ rest] (re-find #"hyperloop_dataset_((\d+_)+)(\D.*)" name)
        factors (keep #(when-not (= % "") (Integer/parseInt %))
                      (str/split factor-str #"_"))
        N (apply * factors)
        d (inc (count factors))
        params (eval-name->params "hyperloop_dataset" rest)]
    (when (:time-eval params)
      (merge params {:full-name name :N N :d d :factors factors}))))

(defn epoch-times
  []
  (let [evals (into []
                    (comp (filter #(str/starts-with? % "hyperloop_dataset"))
                          (keep time-eval-name->params)
                          (keep #(try
                                   (let [times
                                         (json/parse-string (slurp (str "./evaluations/" (:full-name %) "/times.json"))
                                                            true)]
                                     (assoc % :epoch-mean (-> times :summary :mean (* 1000))))
                                   (catch Exception))))
                    (ls-dir "./evaluations/"))
        N-evals (->> evals
                     (filter #(= (:d %) 2))
                     (group-by :N)
                     (into (sorted-map)))
        d-evals (->> evals
                     (filter #(= (:N %) 1024))
                     (group-by :d)
                     (into (sorted-map)))
        row-vals (fn [evals]
                   [(some #(and (= (:name %) "GIN") (:epoch-mean %)) evals)
                    (some #(and (= (:name %) "2-WL-GNN") (= (:r %) 1) (:epoch-mean %)) evals)
                    (some #(and (= (:name %) "2-WL-GNN") (= (:r %) 2) (:epoch-mean %)) evals)
                    (some #(and (= (:name %) "2-WL-GNN") (= (:r %) 3) (:epoch-mean %)) evals)])
        N-head "N,gin,r1,r2,r3"
        d-head "d,gin,r1,r2,r3"
        N-rows (map (fn [[N evals]] (str/join "," (cons N (row-vals evals)))) N-evals)
        d-rows (map (fn [[d evals]] (str/join "," (cons d (row-vals evals)))) d-evals)
        N-csv (str N-head "\n" (str/join "\n" N-rows))
        d-csv (str d-head "\n" (str/join "\n" d-rows))]
    (println (str "Varying N:\n" N-csv "\n"))
    (println (str "Varying d:\n" d-csv))
    (spit (str "../thesis/data/epoch_times_N.csv") N-csv)
    (spit (str "../thesis/data/epoch_times_d.csv") d-csv)))

(defn default-action
  []
  (ds-stats->csv)
  (eval-results->csv {:only-default false} "results")
  (all-fold-differences->tex)
  (wl-depth-compare {:use-gnn false} "wlst_depths")
  (wl-depth-compare {:use-gnn true :pool sam-pool} "wl2_radii_sam")
  (wl-depth-compare {:use-gnn true :pool mean-pool} "wl2_radii_mean"))

(def actions {"ds_stats" ds-stats->csv
              "eval_res_full" (partial eval-results->csv {:only-default false})
              "eval_res" (partial eval-results->csv {:only-default true})
              "fold_diff" (partial fold-differences->tex {:only-default true})
              "fold_diffs" all-fold-differences->tex
              "wlst_compare" (partial wl-depth-compare {:use-gnn false})
              "wl2_compare_sam" (partial wl-depth-compare {:use-gnn true :pool sam-pool})
              "wl2_compare_mean" (partial wl-depth-compare {:use-gnn true :pool mean-pool})
              "durations" durations
              "epoch-times" epoch-times
              nil default-action})

(println "LTAG Results Postprocessor.")
(if-let [action (actions (first *command-line-args*))]
  (apply action (rest *command-line-args*))
  (println "Unknown action:" (first *command-line-args*)))
(println "Done.")
