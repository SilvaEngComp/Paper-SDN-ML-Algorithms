	>???d?L@>???d?L@!>???d?L@	??K??h????K??h??!??K??h??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:>???d?L@g?;p???Ay??L@YO?\?	??rEagerKernelExecute 0*	??n??S@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat}????9??!?p??B@)?P??9??1??+?>@:Preprocessing2U
Iterator::Model::ParallelMapV2?ՏM?#??!3?M???2@)?ՏM?#??13?M???2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?8*7QK??!????$?7@)*?t??1??KT?0@:Preprocessing2F
Iterator::Model??????!??Պ??=@)qW?"???1*?D&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??.4?i??!ΌJ?W?Q@)T?^Pz?1? t ?B @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?ܚt["w?!???5?@)?ܚt["w?1???5?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorB_z?s?p?!?eb/??@)B_z?s?p?1?eb/??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?y7??!???h?9@)??-?[?1?~?? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??K??h??I?????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	g?;p???g?;p???!g?;p???      ??!       "      ??!       *      ??!       2	y??L@y??L@!y??L@:      ??!       B      ??!       J	O?\?	??O?\?	??!O?\?	??R      ??!       Z	O?\?	??O?\?	??!O?\?	??b      ??!       JCPU_ONLYY??K??h??b q?????X@