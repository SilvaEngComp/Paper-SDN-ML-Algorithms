	??9?Ł=@??9?Ł=@!??9?Ł=@	?n? 
???n? 
??!?n? 
??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??9?Ł=@?????r??A??^?G=@Y???U????rEagerKernelExecute 0*	?I+GW@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???hU??!??:7??>@)??#????1????9@:Preprocessing2U
Iterator::Model::ParallelMapV2	À%W???!?5???9@)	À%W???1?5???9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?5[y????!????%6@)q=
ףp??1?1??u?.@:Preprocessing2F
Iterator::Model??q6??!?"??A@)ͮ{+??1??N?p $@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipn??)???!w???{P@)?/J?_???1Xu_T?"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?[X7?y?!?'?W@)?[X7?y?1?'?W@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?m?2dr?!??V??I@)?m?2dr?1??V??I@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?2WՖ?!??O>I?7@)?n??S]?1??K7???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?n? 
??I?Ȗ???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????r???????r??!?????r??      ??!       "      ??!       *      ??!       2	??^?G=@??^?G=@!??^?G=@:      ??!       B      ??!       J	???U???????U????!???U????R      ??!       Z	???U???????U????!???U????b      ??!       JCPU_ONLYY?n? 
??b q?Ȗ???X@