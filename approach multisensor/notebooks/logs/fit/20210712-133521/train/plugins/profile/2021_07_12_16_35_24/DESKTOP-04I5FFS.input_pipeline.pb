	u???a?7@u???a?7@!u???a?7@	8??H???8??H???!8??H???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:u???a?7@?i?L???Aq??ő7@Y	3m??J??rEagerKernelExecute 0*	?????Q@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat"rl=C??!?"Id?@@)ۅ?:????1?B????<@:Preprocessing2U
Iterator::Model::ParallelMapV2?|a2U??!?ϩRB?6@)?|a2U??1?ϩRB?6@:Preprocessing2F
Iterator::Model`?????!1???	B@){?Fw;??1??Q??*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?J̳?V??!?9C??3@)??\5ρ?13?kG?(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip>?x????!?pc??O@)T??b?x?1=??n?? @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?W歺u?!??Y-?@)?W歺u?1??Y-?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor5?\??ul?!?Ij?@)5?\??ul?1?Ij?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapE??S???!'?X?`?5@) ?o_?Y?1???x??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no98??H???I??[|??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?i?L????i?L???!?i?L???      ??!       "      ??!       *      ??!       2	q??ő7@q??ő7@!q??ő7@:      ??!       B      ??!       J		3m??J??	3m??J??!	3m??J??R      ??!       Z		3m??J??	3m??J??!	3m??J??b      ??!       JCPU_ONLYY8??H???b q??[|??X@