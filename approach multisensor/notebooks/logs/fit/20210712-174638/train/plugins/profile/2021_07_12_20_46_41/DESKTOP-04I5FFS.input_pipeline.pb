	֬3?/?B@֬3?/?B@!֬3?/?B@	%U4R????%U4R????!%U4R????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:֬3?/?B@j?*????A+???}?B@Y???O??rEagerKernelExecute 0*	o??ʁS@2U
Iterator::Model::ParallelMapV2J'L5???!?_?Z
i<@)J'L5???1?_?Z
i<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat[(?????!????;@)????kђ?1????B?7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?>???4??!zw?A?8@)?-?????1??ade?1@:Preprocessing2F
Iterator::Model??z???!԰*?$?C@)Ѭl??1~2o}?&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipL?g???!,O?v?N@)??߼8?u?1??y|Av@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??"???s?!?u7?@)??"???s?1?u7?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???$xCj?!?ihdo@)???$xCj?1?ihdo@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapN^d~???!
[}?
?9@)??q???U?1?8??t??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9%U4R????I??V???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	j?*????j?*????!j?*????      ??!       "      ??!       *      ??!       2	+???}?B@+???}?B@!+???}?B@:      ??!       B      ??!       J	???O?????O??!???O??R      ??!       Z	???O?????O??!???O??b      ??!       JCPU_ONLYY%U4R????b q??V???X@