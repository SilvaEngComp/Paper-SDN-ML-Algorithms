	??T?t?2@??T?t?2@!??T?t?2@	wQ#?	???wQ#?	???!wQ#?	???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??T?t?2@?S[? ??A?7M??2@Y?r??+??rEagerKernelExecute 0*	+????Q@2U
Iterator::Model::ParallelMapV2???3???!K?ͯ?T<@)???3???1K?ͯ?T<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatt?Oqx??!?ݬ??z=@)????????1e?Ѥi9@:Preprocessing2F
Iterator::Model?πz3j??!w?a???D@)}!??????1H??c?*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?3Lm????!Ts?:??3@)^f?(?7??1?Wrc*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??̒ 5??!?!?buM@)~?*O ?t?1? ??d?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????|?r?!l!?(?@)????|?r?1l!?(?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??p???g?!??I??C@)??p???g?1??I??C@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???@?m??!?\?Զ?5@)?Д?~PW?1BLGМ @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9wQ#?	???I???T?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?S[? ???S[? ??!?S[? ??      ??!       "      ??!       *      ??!       2	?7M??2@?7M??2@!?7M??2@:      ??!       B      ??!       J	?r??+???r??+??!?r??+??R      ??!       Z	?r??+???r??+??!?r??+??b      ??!       JCPU_ONLYYwQ#?	???b q???T?X@