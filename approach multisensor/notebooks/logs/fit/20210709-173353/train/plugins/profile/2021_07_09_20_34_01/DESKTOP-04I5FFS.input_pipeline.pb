	??#?1@??#?1@!??#?1@	?N??(???N??(??!?N??(??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??#?1@???C???A`??5!?1@Y?U??f???rEagerKernelExecute 0*	?z?G`@2U
Iterator::Model::ParallelMapV2??M(D??!s???.tB@)??M(D??1s???.tB@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeaty#????!l?(?KT5@)EdX????1????1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?"rl??!???^:`6@)???K???1?'g?Wt0@:Preprocessing2F
Iterator::ModelL???!??!I???G@)??@?9w??1T?GV?$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?N???N??!?i??SJ@)?,??;???1;(TGD
@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice@?5_%?!??	v??@)@?5_%?1??	v??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??j?q?!???|?@)??j?q?1???|?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap;M?O??!?<??<?7@)U???)^?1?[N)???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?N??(??I?bX??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???C??????C???!???C???      ??!       "      ??!       *      ??!       2	`??5!?1@`??5!?1@!`??5!?1@:      ??!       B      ??!       J	?U??f????U??f???!?U??f???R      ??!       Z	?U??f????U??f???!?U??f???b      ??!       JCPU_ONLYY?N??(??b q?bX??X@