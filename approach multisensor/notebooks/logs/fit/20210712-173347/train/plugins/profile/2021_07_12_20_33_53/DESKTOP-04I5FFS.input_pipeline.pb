	2U0*???@2U0*???@!2U0*???@	ץ????ץ????!ץ????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:2U0*???@AG?Z???A?>????@YL?4??a??rEagerKernelExecute 0*	?ZdS]@2U
Iterator::Model::ParallelMapV2??%䃞??!?????A@)??%䃞??1?????A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?zk`???!lOaE7~6@)?<ڨN??1?].Rg3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??M??ܚ?!A"I/]6@)#??Jvl??1B???? 1@:Preprocessing2F
Iterator::ModelU?3?Y??!???F@)??J?????1hM,??#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?׼??Z??!?;?=?:K@)?3??k???1?)??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?I?%r?y?!???? q@)?I?%r?y?1???? q@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor[?a/?m?!????)?@)[?a/?m?1????)?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?IEc????!???18@)@??wԘ`?1Q??$i???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9إ????I-1~v?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	AG?Z???AG?Z???!AG?Z???      ??!       "      ??!       *      ??!       2	?>????@?>????@!?>????@:      ??!       B      ??!       J	L?4??a??L?4??a??!L?4??a??R      ??!       Z	L?4??a??L?4??a??!L?4??a??b      ??!       JCPU_ONLYYإ????b q-1~v?X@