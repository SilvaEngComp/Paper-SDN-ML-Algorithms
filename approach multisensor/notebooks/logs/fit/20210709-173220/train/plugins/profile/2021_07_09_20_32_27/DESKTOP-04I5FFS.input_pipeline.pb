	z?"n?1@z?"n?1@!z?"n?1@	??*??????*????!??*????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:z?"n?1@???n-??A?0E?4V1@Y62;?޽?rEagerKernelExecute 0*	D?l???d@2U
Iterator::Model::ParallelMapV2?g\W̨?!;?/?W=@)?g\W̨?1;?/?W=@:Preprocessing2F
Iterator::Modelɬ??vh??!?	C\?%H@)???g???1??u??2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[A?+??!|?%qa?6@)??A?p???1?O??α1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatT?????!??䘒3@)?U??f???1%p???0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?I??	ٵ?!f????I@)?2??bb??1Y??y??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceGˁjۀ?!%??-J?@)Gˁjۀ?1%??-J?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??(?N??!??E?e:@)AgҦ?y?1?X?~?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??'??q?!?݃Pi @)??'??q?1?݃Pi @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??*????I?yj??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???n-?????n-??!???n-??      ??!       "      ??!       *      ??!       2	?0E?4V1@?0E?4V1@!?0E?4V1@:      ??!       B      ??!       J	62;?޽?62;?޽?!62;?޽?R      ??!       Z	62;?޽?62;?޽?!62;?޽?b      ??!       JCPU_ONLYY??*????b q?yj??X@