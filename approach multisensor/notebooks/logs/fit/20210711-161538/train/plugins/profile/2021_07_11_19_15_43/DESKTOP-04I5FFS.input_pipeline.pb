	?N??:;=@?N??:;=@!?N??:;=@	r?y???r?y???!r?y???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?N??:;=@????q???A^H??0?<@Y'f?ʉ??rEagerKernelExecute 0*	u?V@r@2U
Iterator::Model::ParallelMapV2?N@a÷?!?E?8??@)?N@a÷?1?E?8??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??gx???!????>@)??)?D/??1К???9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?)?n???!???n7@)?? ?X4??1e>Rh?3@:Preprocessing2F
Iterator::ModelMjh???!?Qg?fC@)?n??;???1Cx=U6@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip.?????!???y?N@)?????1?F0??y@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?@?9w???!????@)?@?9w???1????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???DR??!3(???1@)???DR??13(???1@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaptB??K8??!2???2@@)???s?1??LE????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9r?y???I??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????q???????q???!????q???      ??!       "      ??!       *      ??!       2	^H??0?<@^H??0?<@!^H??0?<@:      ??!       B      ??!       J	'f?ʉ??'f?ʉ??!'f?ʉ??R      ??!       Z	'f?ʉ??'f?ʉ??!'f?ʉ??b      ??!       JCPU_ONLYYr?y???b q??X@