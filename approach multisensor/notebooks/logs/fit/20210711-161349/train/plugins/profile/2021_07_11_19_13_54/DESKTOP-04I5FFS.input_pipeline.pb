	?|x? ?,@?|x? ?,@!?|x? ?,@	%??~????%??~????!%??~????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?|x? ?,@
H?`???A<?D?+@Y??{?&??rEagerKernelExecute 0*	A`??"W@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??????!?]r??SG@)?U??Ά??1r?ϷE@:Preprocessing2U
Iterator::Model::ParallelMapV2)??????!?< ??6@))??????1?< ??6@:Preprocessing2F
Iterator::ModelA*Ŏơ??!?j?w4@@)Tb.???1I?:?"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate5A?} R??!t??=??,@)?o??e1??1?G?|?0"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?uT5At?!?İ?+n@)?uT5At?1?İ?+n@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???=????!?ʘp??P@)???-?r?1???m?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?ۼqRh?!???x׻	@)?ۼqRh?1???x׻	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapr6?,??!jR?r:?/@)??G??V?1????9*??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9%??~????Il?}?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	
H?`???
H?`???!
H?`???      ??!       "      ??!       *      ??!       2	<?D?+@<?D?+@!<?D?+@:      ??!       B      ??!       J	??{?&????{?&??!??{?&??R      ??!       Z	??{?&????{?&??!??{?&??b      ??!       JCPU_ONLYY%??~????b ql?}?X@