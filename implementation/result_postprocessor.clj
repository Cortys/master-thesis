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
                  "DD" "dd"
                  "REDDIT-BINARY" "reddit"
                  "IMDB-BINARY" "imdb"})
(def datasets ["balanced_triangle_classification_dataset"
               "NCI1" "PROTEINS_full" "DD" "REDDIT-BINARY" "IMDB-BINARY"])

(defn round
  ([num] (round num 0))
  ([num prec]
   (if (int? num)
     num
     (if (zero? prec)
       (int (Math/round num))
       (let [p (Math/pow 10 prec)]
         (/ (Math/round (* num p)) p))))))

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
             (+ dim_node_features num_node_labels)
             (+ dim_edge_features num_edge_labels)
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
    (if (str/ends-with? name "FC")
      (str pool "+ MLP")
      pool)))

(defn eval-name->params
  [dataset name single-depth]
  (let [[_ r n _ T] (re-find #"n?(\d?)_?(.+?)(_FC)?(_\d)?$" name)
        r (if (seq r) (Integer/parseInt r) (default-radii dataset))
        T (when (seq T) (Integer/parseInt (subs T 1)))
        pool (extract-pool extract-pool)]
    (condp #(str/includes? %2 %1) n
      "CWL2GCN"
      (when (or (not single-depth)
                (= r (single-depth-radii dataset)))
        {:name "2-WL-GNN"
         :order 14
         :radius (when (not single-depth) (str "r=" r))
         :pool pool})
      "K2GNN"
      {:name "2-GNN"
       :order 13
       :pool pool}
      "WL_st"
      (when (or (not single-depth) (nil? T))
        {:name "WL\\textsubscript{ST}"
         :order 1
         :it (str "T=" (or T 5))})
      "WL_sp"
      (when (or (not single-depth) (nil? T))
        {:name "WL\\textsubscript{SP}"
         :order 2
         :it (str "T=" (or T 5))})
      "LWL2"
      (when (or (not single-depth) (nil? T))
        {:name "2-LWL"
         :order 3
         :it (str "T=" (or T 3))})
      "GWL2"
      (when (or (not single-depth) (nil? T))
        {:name "2-GWL"
         :order 4
         :it (str "T=" (or T 3))})
      "GIN"
      {:name "GIN" :order 12}
      "MolecularFingerprint"
      {:name "Baseline" :order 11}
      "DeepMultisets"
      {:name "Baseline" :order 11}
      nil)))

(defn dataset-results
  [dataset & {:keys [single-depth] :or {single-depth true}}]
  (let [{evals :out} (sh "ls" "./evaluations/")
        evals (str/split evals #"\n+")
        summaries (into []
                        (comp (filter #(and (str/starts-with? % dataset) (not (str/ends-with? % "quick"))))
                              (map (juxt identity #(try (slurp (str "./evaluations/" % "/summary/results.json")) (catch Exception _))))
                              (keep (fn [[name s]] (when s (assoc (json/parse-string s true)
                                                                  :name (subs name (inc (count dataset)))))))
                              (keep (fn [{name :name
                                          {test :accuracy} :combined_test
                                          {train :accuracy} :combined_train}]
                                      {:name name
                                       :test-mean (:mean test)
                                       :test-std (:std test)
                                       :train-mean (:mean train)
                                       :train-std (:std train)})))
                        evals)
        {comp-evals :out} (sh "ls" "./libs/gnn-comparison/RESULTS/")
        comp-evals (str/split comp-evals #"\n+")
        summaries (into summaries
                        (comp (filter #(str/ends-with? % (str dataset "_assessment")))
                              (map (juxt identity #(try
                                                     (slurp (str "./libs/gnn-comparison/RESULTS/" % "/10_NESTED_CV/assessment_results.json"))
                                                     (catch Exception e (println e)))))
                              (keep (fn [[name s]] (when s (assoc (json/parse-string s true)
                                                                  :name name))))
                              (keep (fn [{:keys [name avg_TR_score std_TR_score avg_TS_score std_TS_score]}]
                                      {:name name
                                       :test-mean avg_TS_score
                                       :test-std std_TS_score
                                       :train-mean avg_TR_score
                                       :train-std std_TR_score})))
                        comp-evals)
        summaries (keep (fn [{name :name :as sum}]
                           (when-let [params (eval-name->params dataset name single-depth)]
                             (merge sum {:model (params :name)
                                         :order (params :order)
                                         :dataset dataset
                                         :params (str/join ", " (keep params [:pool :radius :it]))})))
                        summaries)]
      summaries))

(defn dataset-result-head
  [dataset]
  (let [pre (ds-colnames dataset)]
    (str pre "TestMean;" pre "TestStd;" pre "TrainMean;" pre "TrainStd")))

(defn dataset-result-row
  [datasets results [model params]]
  (let [results (into {}
                      (comp (filter #(and (= (:model %) model) (= (:params %) params)))
                            (map (juxt :dataset identity)))
                      results)]
    (str model ";" params ";"
         (str/join ";" (map (fn [ds]
                              (if-let [res (results ds)]
                                (str/join ";" (->> [:test-mean :test-std :train-mean :train-std]
                                                   (map res)))
                                                   ; (map #(* 100 %))))
                                "-;-;-;-"))
                            datasets)))))

(defn eval-results->csv
  [{:keys [single-depth]} & args]
  (let [results (sort-by :order (mapcat #(dataset-results % :single-depth single-depth) (if (empty? args) datasets args)))
        models-with-params (distinct (map (juxt :model :params) results))
        head (str "model;params;" (str/join ";" (map dataset-result-head datasets)))
        rows (into [] (map (partial dataset-result-row datasets results)) models-with-params)
        csv (str head "\n" (str/join "\n" rows))]
    (println csv)))

(def actions {"ds_stats" ds-stats->csv
              "eval_res_full" (partial eval-results->csv {:single-depth false})
              "eval_res" (partial eval-results->csv {:single-depth true})})

(println "LTAG Results Postprocessor.")
(if-let [action (actions (first *command-line-args*))]
  (apply action (rest *command-line-args*))
  (println "Unknown action:" (first *command-line-args*)))
(println "Done.")
