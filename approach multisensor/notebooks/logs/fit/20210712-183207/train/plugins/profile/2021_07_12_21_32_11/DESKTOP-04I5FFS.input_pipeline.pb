	w?*2*O@w?*2*O@!w?*2*O@	??G?|L????G?|L??!??G?|L??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:w?*2*O@z9??cx??A;S???O@Y?m?8)̫?rEagerKernelExecute 0*	a??"?)R@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?9τ&??!|F?u?@@)5s?????1ҋ?34<@:Preprocessing2U
Iterator::Model::ParallelMapV2;??]؊?!?/?W?
2@);??]؊?1?/?W?
2@:Preprocessing2F
Iterator::Model??\??k??!???W?A@)?;2V????1y	@W? 0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateK[\?3ُ?!l?A?xg5@)?d?VA??1???g?*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???_?|??!?19?uP@)U1?~?y?1e?-??O!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?????w?!X(???@)?????w?1X(???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??PN??p?!?l?Vh@)??PN??p?1?l?Vh@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?K?b??!BN?|H^7@)ˆ5?EaW?1T?y$?l??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??G?|L??I???l?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	z9??cx??z9??cx??!z9??cx??      ??!       "      ??!       *      ??!       2	;S???O@;S???O@!;S???O@:      ??!       B      ??!       J	?m?8)̫??m?8)̫?!?m?8)̫?R      ??!       Z	?m?8)̫??m?8)̫?!?m?8)̫?b      ??!       JCPU_ONLYY??G?|L??b q???l?X@