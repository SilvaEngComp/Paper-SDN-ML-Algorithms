	???y?B@???y?B@!???y?B@	??۝??????۝????!??۝????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:???y?B@?????_??A1??f?A@Y?f?\S ??rEagerKernelExecute 0*	?&1??Q@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??u?X??!Ÿ????@@)?i?q????1#U???<@:Preprocessing2U
Iterator::Model::ParallelMapV2?72?????!?x܄P7@)?72?????1?x܄P7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate&??[X7??!K???5@)_~?Ɍ???1©?16.@:Preprocessing2F
Iterator::Modele??]????!%3??d?A@)??f??}??1?ڗĂU(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipv?U?0??!n?5?M!P@)֬3?/.u?1 ??w@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?St$??p?!?9`ջ?@)?St$??p?1?9`ջ?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor'???Sn?!?q??8@)'???Sn?1?q??8@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?{?E{???!?A?_H7@)[|
??Z?1??B??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??۝????I(????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????_???????_??!?????_??      ??!       "      ??!       *      ??!       2	1??f?A@1??f?A@!1??f?A@:      ??!       B      ??!       J	?f?\S ???f?\S ??!?f?\S ??R      ??!       Z	?f?\S ???f?\S ??!?f?\S ??b      ??!       JCPU_ONLYY??۝????b q(????X@