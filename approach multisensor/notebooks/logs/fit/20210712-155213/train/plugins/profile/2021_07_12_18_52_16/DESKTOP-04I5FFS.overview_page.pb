?	q???98@q???98@!q???98@	~n????~n????!~n????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:q???98@?2?FY??A@h=|??7@Y??(?'???rEagerKernelExecute 0*	?V]J@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?<Fy????!U?~L?@)?'??Q??1|jLc?9:@:Preprocessing2U
Iterator::Model::ParallelMapV2g|_\?҆?!4?ҡ"5@)g|_\?҆?14?ҡ"5@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip$c??ա?!?>	???P@)T:X??0?18O?@w?,@:Preprocessing2F
Iterator::Model?4S??!???0?@@)8?q???{?1????|?)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate\?M4???!?r???+2@)7¢"N'y?1?~R'K'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???3.l?!??u?@)???3.l?1??u?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorj?????e?!gSʈI@)j?????e?1gSʈI@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapo???????!7?7XR4@)?xy:W?R?1;??4@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9}n????Is??91?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?2?FY???2?FY??!?2?FY??      ??!       "      ??!       *      ??!       2	@h=|??7@@h=|??7@!@h=|??7@:      ??!       B      ??!       J	??(?'?????(?'???!??(?'???R      ??!       Z	??(?'?????(?'???!??(?'???b      ??!       JCPU_ONLYY}n????b qs??91?X@Y      Y@q?fb?@"?
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