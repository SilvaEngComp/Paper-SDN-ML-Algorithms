?	??'???1@??'???1@!??'???1@	qN????qN????!qN????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??'???1@???E???A?|[?T?1@Y2Xq??0??rEagerKernelExecute 0*	?C?l??W@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?_>Y1\??!??v{{>@)????????1TD?:@:Preprocessing2U
Iterator::Model::ParallelMapV2?w?~?~??!??4?Ux:@)?w?~?~??1??4?Ux:@:Preprocessing2F
Iterator::Model???fd???!R?y]?OD@)?????C??1g|1?N,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate%?????!?ǭ?r@4@)(~??k	??1???g?)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceD??{?{?!??к?@)D??{?{?1??к?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipF??_???!?|??,?M@)N^??y?1H?U??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor
?s34n?!??c?[@)
?s34n?1??c?[@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapF?7?k??!@?P_=6@)?x'?^?1??+Z????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9qN????I??4?
?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???E??????E???!???E???      ??!       "      ??!       *      ??!       2	?|[?T?1@?|[?T?1@!?|[?T?1@:      ??!       B      ??!       J	2Xq??0??2Xq??0??!2Xq??0??R      ??!       Z	2Xq??0??2Xq??0??!2Xq??0??b      ??!       JCPU_ONLYYqN????b q??4?
?X@Y      Y@qL?W,??@"?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 