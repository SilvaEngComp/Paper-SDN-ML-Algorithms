	?? d?@@?? d?@@!?? d?@@	?,?4I???,?4I??!?,?4I??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?? d?@@?R	??A?)?:?@@Y3k) ???rEagerKernelExecute 0*	i??|??U@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatU?????!'I???,C@)Z??mē??1Ȕ:J?@@:Preprocessing2U
Iterator::Model::ParallelMapV2m??~????!?e?(?2@)m??~????1?e?(?2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???"???!??%H?4@)??Fu:???1Yu9K[(@:Preprocessing2F
Iterator::ModelD3O?)???!???9??<@)???^Dہ?13|ߌh+$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??ݯ|??!Ɋ??Q@)?'Hlw??1ɺ?$"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceצ???~?!ð?E{!@)צ???~?1ð?E{!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor¾?D?q?!??????@)¾?D?q?1??????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??*????!??Tɖ?7@)??
?c?1?}?!uJ@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?,?4I??Ijq?e??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?R	???R	??!?R	??      ??!       "      ??!       *      ??!       2	?)?:?@@?)?:?@@!?)?:?@@:      ??!       B      ??!       J	3k) ???3k) ???!3k) ???R      ??!       Z	3k) ???3k) ???!3k) ???b      ??!       JCPU_ONLYY?,?4I??b qjq?e??X@