?	NG 7?eB@NG 7?eB@!NG 7?eB@	??gз????gз??!??gз??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:NG 7?eB@7?C???Ag??eMB@Y?n????rEagerKernelExecute 0*	?MbX?T@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat֩?=#??!Gw??#0E@)\Ɏ?@???1/?M?B@:Preprocessing2U
Iterator::Model::ParallelMapV20????q??!?k_??5@)0????q??1?k_??5@:Preprocessing2F
Iterator::Model?l??爜?!gw??k?@@)??#F?-??1y??6?'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?W?މ?!??/?V.@)b?[>??~?1??|P?!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??N^??!LD	J?P@)x???Ĭw?1??????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorA)Z??u?!Ű?;S@)A)Z??u?1Ű?;S@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???o
+u?!?]e,1?@)???o
+u?1?]e,1?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapk??=]??!g?vD?71@)D??{?[?1?k???c @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??gз??IL?$?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	7?C???7?C???!7?C???      ??!       "      ??!       *      ??!       2	g??eMB@g??eMB@!g??eMB@:      ??!       B      ??!       J	?n?????n????!?n????R      ??!       Z	?n?????n????!?n????b      ??!       JCPU_ONLYY??gз??b qL?$?X@Y      Y@q???M??"?
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