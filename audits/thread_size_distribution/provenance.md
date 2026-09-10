# Provenance

## Version anchors

- Scripts Git repository: `/home/cara/Documents/reddit_analyses/thread-size/Scripts`
- Scripts commit: `1051e881a2d0fe4a9ae7886d2d3b2985d0da610a`
- Scripts working-tree note at audit time: prior audit directories and this new audit package are untracked; no modelling or thesis output was modified.
- Data archive metadata: `Zenodo_upload/README.md` identifies archive version 1.0 and DOI `10.5281/zenodo.17831100`.
- Raw `Inputs/*_threads.parquet` files were separately checked as byte-identical to the corresponding `Zenodo_upload/data_raw/*_threads.parquet` files.
- Generated UTC: `2026-09-10T19:30:28.064748+00:00`

## Authority chain

The executed preprocessing logs identify `Inputs/{subreddit}_threads.parquet` and `Inputs/{subreddit}_comments.parquet` as feature-construction inputs. The feature output is chronologically split 80/20 by `2_tf_idf_analysis.py`; the split rows are retained in `*_svd_enriched_{train,test}_data.parquet`; and `3_model_data.py` carries their outcomes into the final `*_{train,test}_Y.parquet` files. Those final outcome files control the distribution audit. Raw comments are used only to verify the `T=1+C` identity and self-comment treatment. Selected-model workbooks and the publication ratio workbook are reconciliation targets, not sources used to define the distributions.

## SHA-256 source inventory

| Path                                                                                 | Role                                         | SHA-256                                                          |
| ------------------------------------------------------------------------------------ | -------------------------------------------- | ---------------------------------------------------------------- |
| Inputs/conspiracy_comments.parquet                                                   | raw retained descendant-comment records      | 5c66d6ad9e9e03ad1ddabd25d3b162c6269b8d18929793bf8e852de42afd042e |
| Inputs/conspiracy_threads.parquet                                                    | raw eligible root records                    | 8154462f14cf2b12a4e913a068828be9a73bc6d6eb9619fae17ac54a4a1040ca |
| Inputs/crypto_comments.parquet                                                       | raw retained descendant-comment records      | c492bf6d326b638385d3138ff68b5b0a2c09e8f500f6ab2ddf7866c269266f9d |
| Inputs/crypto_threads.parquet                                                        | raw eligible root records                    | 06655d9490fae4063f3ab76f22c6103c6ac3a94940feeb7ce4c82790c167e185 |
| Inputs/politics_comments.parquet                                                     | raw retained descendant-comment records      | 2d3f4fff24040aa7dc3648d55428a471c0b8bf1d496002cdfa05f87b21b122c0 |
| Inputs/politics_threads.parquet                                                      | raw eligible root records                    | 43f055ccafc8f1c1cb70dfe93f5174c91259fc990e852a5e668f5ca13053ddff |
| Outputs/0_preprocessing/conspiracy/3_model_data_log.xlsx                             | final preprocessing population table         | 5e2bf999102aca354ddeb1f2b67ff6efd794910885566c038d30d64b4b467445 |
| Outputs/0_preprocessing/conspiracy/conspiracy_test_Y.parquet                         | final chronological held-out outcomes        | 1fc9d302c1cf5b1353dd27b762c9ee45c3a3673a464b174e3965c621a2055d71 |
| Outputs/0_preprocessing/conspiracy/conspiracy_threads_extra_feats.parquet            | feature-construction root output             | af6e749a2e12e8839e56ecaddff3fa41beb02c0babf6514a40a5b843d925bfca |
| Outputs/0_preprocessing/conspiracy/conspiracy_train_Y.parquet                        | final training outcomes                      | acad63a61e7dfb16069111b074e5fa00e4f10cde6cf9aaddac7acbb1fdb8a62e |
| Outputs/0_preprocessing/conspiracy/logs/conspiracy_0_1_construct_features.out        | executed preprocessing log                   | c7553234c364cdba9e1ae9bb034cfece19e360148f95ca9ea153efe595ffdad7 |
| Outputs/0_preprocessing/conspiracy/logs/conspiracy_0_2_tfidf_analysis.out            | executed preprocessing log                   | ec26cbb14f56a8f67030df9a76217a1defe7bbc0610a3420d46e063fc3462568 |
| Outputs/0_preprocessing/conspiracy/logs/conspiracy_0_3_model_data.out                | executed preprocessing log                   | b1604aae1c0fd957f073be2b467885c3d21fa1a22522f69b589a6bed854330e1 |
| Outputs/0_preprocessing/conspiracy/tf-idf/conspiracy_svd_enriched_test_data.parquet  | held-out row-to-thread mapping               | 7c09c0101fd6df791e7f3adf2088120a24fa851001fe4f3e9c1f2e8833815942 |
| Outputs/0_preprocessing/conspiracy/tf-idf/conspiracy_svd_enriched_train_data.parquet | training row-to-thread mapping               | 4d53bcf901477bd91a88caf9794aa5c2d0914fd20db6deb6067889d7aa2f5cdd |
| Outputs/0_preprocessing/crypto/3_model_data_log.xlsx                                 | final preprocessing population table         | 7af28f8b9faa87fe88c227abc939be7dbeef2c89ae31c70e55744c5785da2676 |
| Outputs/0_preprocessing/crypto/crypto_test_Y.parquet                                 | final chronological held-out outcomes        | 27291ce72204717b7df8cee66dede6986738936b8e2832bb36939a895341b5d3 |
| Outputs/0_preprocessing/crypto/crypto_threads_extra_feats.parquet                    | feature-construction root output             | 9c8848f2f41254feaae68d7d700a6772fa5ad639dc658ff8c84d041d00fb1edb |
| Outputs/0_preprocessing/crypto/crypto_train_Y.parquet                                | final training outcomes                      | 38dc3fc74b9a84245a68938116506617c23746736f8391e8c7fc9c6d3289311f |
| Outputs/0_preprocessing/crypto/logs/crypto_0_1_construct_features.out                | executed preprocessing log                   | 1b0bcce353f028ee2c2c18c9fa388c2dd7dc075cfd14b3bc1180b5fd85b89a94 |
| Outputs/0_preprocessing/crypto/logs/crypto_0_2_tfidf_analysis.out                    | executed preprocessing log                   | 82f943b54af7a82985441e63125513ef84dfa198a014ae8275df0f5ead3d235d |
| Outputs/0_preprocessing/crypto/logs/crypto_0_3_model_data.out                        | executed preprocessing log                   | 0b3433734d4b88a96174f6a548be8a18ac926b8ef75dd48239a5facf2c8753d2 |
| Outputs/0_preprocessing/crypto/tf-idf/crypto_svd_enriched_test_data.parquet          | held-out row-to-thread mapping               | 179ee6077476fa4878b46774be40378ceba8867d94723f00b658b49302061233 |
| Outputs/0_preprocessing/crypto/tf-idf/crypto_svd_enriched_train_data.parquet         | training row-to-thread mapping               | 435061ded64cf05de5706b4c597e2a7f78fd080df9b8f6322a4ab876d5ecab6c |
| Outputs/0_preprocessing/politics/3_model_data_log.xlsx                               | final preprocessing population table         | 78d83a59e6f2ca774415c6d4dac925db5c0027657b1e30f5f871be4da7d3c666 |
| Outputs/0_preprocessing/politics/logs/politics_0_1_construct_features.out            | executed preprocessing log                   | 68efb91d66cb11ef0aa7a06a182d81e5b67a29c9079fff7c735d9e429df315d7 |
| Outputs/0_preprocessing/politics/logs/politics_0_2_tfidf_analysis.out                | executed preprocessing log                   | cb573d4654977073bba11af0b58b1318d52ea7367659db0adbfd8656b48e5363 |
| Outputs/0_preprocessing/politics/logs/politics_0_3_model_data.out                    | executed preprocessing log                   | e44b722378973b337b88a3be1f7f8947b9b8cb0591acc301ea0226aa95bcbd68 |
| Outputs/0_preprocessing/politics/politics_test_Y.parquet                             | final chronological held-out outcomes        | 4e164854f234b5fda8f9c50decb3e2cd94afd2cf1ddf4be40f42a9a15964f5c8 |
| Outputs/0_preprocessing/politics/politics_threads_extra_feats.parquet                | feature-construction root output             | 9ceedfc958183659547cf531b78d221ad30ed481c410db9d5814bf6d2e8727ab |
| Outputs/0_preprocessing/politics/politics_train_Y.parquet                            | final training outcomes                      | 1016feb57b822593c93244efb0203cc2197c9bffa896493ffad89dbc19bd0d4e |
| Outputs/0_preprocessing/politics/tf-idf/politics_svd_enriched_test_data.parquet      | held-out row-to-thread mapping               | 8088ee012fd253f32bc07fea6e5fd01bde05398cdefa577bda04ef5e6b8c77a6 |
| Outputs/0_preprocessing/politics/tf-idf/politics_svd_enriched_train_data.parquet     | training row-to-thread mapping               | e112a97f58be946d55483025b03d79b287c7fd0b579a14e1258f11b8e121da03 |
| Outputs/2_thread_size/conspiracy/3_h_tuning/params_post_hyperparam_tuning.jl         | frozen training-derived class boundaries     | 35db3d661610579d446bed1e87b1454c987da00f00f2ab61eafca8aac525fc2c |
| Outputs/2_thread_size/conspiracy/4_model/model_3/test_data_results.xlsx              | selected-model class-distribution table      | 948da32f6b2a96c4115b89698df7b9d6b23b97cbb5e328d72ce0690b663aec09 |
| Outputs/2_thread_size/crypto/3_h_tuning/params_post_hyperparam_tuning.jl             | frozen training-derived class boundaries     | 52ad2be022b730259b26d3b2a6e754615a489481f0dd71824e5ef46f4ac607d7 |
| Outputs/2_thread_size/crypto/4_model/model_2/test_data_results.xlsx                  | selected-model class-distribution table      | e0eb3aae582e68ebca1a9158c6dd8d41b40e36afbf1a7a3121ca7ded117a8ff9 |
| Outputs/2_thread_size/politics/3_h_tuning/params_post_hyperparam_tuning.jl           | frozen training-derived class boundaries     | a9c671e5fe659a611d4c38c0814167256e450973fde7d9425c49bf19186f071d |
| Outputs/2_thread_size/politics/4_model/model_3/test_data_results.xlsx                | selected-model class-distribution table      | e11f23a7df0d31794186524511f2bbbf5926404cd368d1436b79df20de9890a1 |
| Publication_Outputs/2_Thread_Size/cms/predicted_class_ratios.xlsx                    | thesis-facing true class-distribution ratios | 6fa286edff8f6f746d7ee762cfc481cfee20bf770c8cebc9a7a2de19d578235b |
| Scripts/0_Preprocessing/1_construct_features.py                                      | code or prior validation evidence            | e33a79e89e0132f149ffddc1abb675583228d4c8892b0066e1d7d05fdbe9bd81 |
| Scripts/0_Preprocessing/2_tf_idf_analysis.py                                         | code or prior validation evidence            | a1e437a59d160293029a9cc0a01f65e17fcc38e23e26855433c86da272861eae |
| Scripts/0_Preprocessing/3_model_data.py                                              | code or prior validation evidence            | c463fbd8b503d6e912591fef508091e7502fbfa8082bd8766417305bbc88f434 |
| Scripts/2_Thread_size/2_tuning.py                                                    | code or prior validation evidence            | 90034f98687501b9b48e7c4d5e6f7d585324fc7b3765f7c2c4ffa6669d538d17 |
| Scripts/2_Thread_size/3_hyperparameter_tuning.py                                     | code or prior validation evidence            | 0b9ea1fc016ff399d541acc0da75033ebf077087bb189b60270138ad066d2bf7 |
| Scripts/2_Thread_size/4_run_tuned_model.py                                           | code or prior validation evidence            | 252c434551e2bd6c092b44625690772ce261bc0716b8209dddc3d72ee9c1f518 |
| Scripts/audits/probability_calibration/outputs/chronological_partitions.csv          | code or prior validation evidence            | acc6de35ae31e5bdb8e0d6160dd18c711564c5feb7643bc07d5c039de61eb221 |
| Scripts/audits/probability_calibration/outputs/validation_checks.csv                 | code or prior validation evidence            | 65b84ed8164b5a511ab2377654ddbebd5653c86ceb1d6ee32ebb8d18708cdd02 |
| Scripts/audits/thread_size_distribution/audit_thread_size_distribution.py            | code or prior validation evidence            | 39bc1ab2e6c187b736b8c51d9a42f07890c63531e9f780a3e4e54380a814ef11 |
| Scripts/audits/thread_size_distribution/render_ccdf.py                               | code or prior validation evidence            | 4b79fed2b66802a8638e49dc02defc0b954c547b01809774bd2a9514cb35a947 |

The machine-readable inventory also records byte sizes, modification times and version bases in `outputs/source_inventory.csv`. Generated aggregate outputs are reproducible from the named sources; the report and figures are not inputs to any modelling pipeline.
