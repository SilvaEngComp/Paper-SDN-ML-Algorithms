	Ig`?e[G@Ig`?e[G@!Ig`?e[G@	O???J??O???J??!O???J??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:Ig`?e[G@{???`???A?5??>G@Y?1zn???rEagerKernelExecute 0*	?G?z[@2U
Iterator::Model::ParallelMapV2?n?燩?!???b?	G@)?n?燩?1???b?	G@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatW?Sb??!???`?";@)k???#G??1?|?z?7@:Preprocessing2F
Iterator::Model?l\???!C?????K@)e???݅?1TJ?$f?#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??aMeQ??!??J??%@)+i?7>{?1?=mIP?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??b?du?!?]?JN@)??b?du?1?]?JN@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipP5z5@i??!?p=F@)?d#?#t?1.?v&?,@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor`???Yn?!W?
?~c@)`???Yn?1W?
?~c@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap6t??Pn??!׹8???(@)???W?X?1H`?y??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9P???J??I3r$?Z?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	{???`???{???`???!{???`???      ??!       "      ??!       *      ??!       2	?5??>G@?5??>G@!?5??>G@:      ??!       B      ??!       J	?1zn????1zn???!?1zn???R      ??!       Z	?1zn????1zn???!?1zn???b      ??!       JCPU_ONLYYP???J??b q3r$?Z?X@