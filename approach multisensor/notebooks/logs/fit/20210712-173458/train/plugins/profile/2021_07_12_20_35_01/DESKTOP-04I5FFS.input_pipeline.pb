	?ګ?{?@?ګ?{?@!?ګ?{?@	?&???????&??????!?&??????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?ګ?{?@????W??A? 4J?B?@YʋL?????rEagerKernelExecute 0*	A5^?I?O@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??@?ȓ?!?eۼhp>@)???I???1\?+??:@:Preprocessing2U
Iterator::Model::ParallelMapV2?CmFA??!??U?9@)?CmFA??1??U?9@:Preprocessing2F
Iterator::Model?R]????!?I??;?D@)??I???1?c???0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate Q?????!f[D?r3@)Wzm6Vb~?1?"???_'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??4s?!?(?QV@)??4s?1?(?QV@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?zNz????!i?
`?0M@)?;P?<?q?1D?)̓F@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorg|_\??f?!?]?~??@)g|_\??f?1?]?~??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?L!u??!B??5@)? ݗ3?U?1?~Y?`? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?&??????I?? %??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????W??????W??!????W??      ??!       "      ??!       *      ??!       2	? 4J?B?@? 4J?B?@!? 4J?B?@:      ??!       B      ??!       J	ʋL?????ʋL?????!ʋL?????R      ??!       Z	ʋL?????ʋL?????!ʋL?????b      ??!       JCPU_ONLYY?&??????b q?? %??X@