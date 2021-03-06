?	?d??7?B@?d??7?B@!?d??7?B@	f:?a#k??f:?a#k??!f:?a#k??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?d??7?B@?}?e????A??ɍ"kB@Yr?????rEagerKernelExecute 0*	?MbX?R@2U
Iterator::Model::ParallelMapV2?k?????!?s?_=@)?k?????1?s?_=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat4?i?????!??<??=@)????????1"V? ??9@:Preprocessing2F
Iterator::Model?乾??!<?$?F@)?D?????1၅Z?x-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate6??gΊ?!S???\1@)??F?????1H????'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???5>???!?e?&??K@)Q?|a2u?1?A?T}t@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??q?d?p?!,-??s@)??q?d?p?1,-??s@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorl??g?i?!?N
i??@)l??g?i?1?N
i??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap75?|΍?!-QG]?M3@)G?ŧ X?1?:????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9f:?a#k??IcOn??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?}?e?????}?e????!?}?e????      ??!       "      ??!       *      ??!       2	??ɍ"kB@??ɍ"kB@!??ɍ"kB@:      ??!       B      ??!       J	r?????r?????!r?????R      ??!       Z	r?????r?????!r?????b      ??!       JCPU_ONLYYf:?a#k??b qcOn??X@Y      Y@q?8?"3???"?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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