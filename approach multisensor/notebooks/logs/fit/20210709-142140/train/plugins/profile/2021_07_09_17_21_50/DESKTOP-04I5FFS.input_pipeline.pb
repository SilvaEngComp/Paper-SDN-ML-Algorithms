	??,??j4@??,??j4@!??,??j4@	??H?0????H?0??!??H?0??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??,??j4@BC?+??A?
???#4@Y???M?q??rEagerKernelExecute 0*	? ?rh1a@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?:????!
??y<@)O>=?e???1???z59@:Preprocessing2U
Iterator::Model::ParallelMapV28???????!$???3@)8???????1$???3@:Preprocessing2F
Iterator::Modeln??)"??!???I?CC@)?7?GnM??1m	/???2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????n??!Q|?K??4@)??m3???1?????d)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?{?q??!?Bg??f @)?{?q??1?Bg??f @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipA??? ???!Iiy?9?N@)??5>????1???b@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?E	?=??!????9@)??w?'-|?1(?W@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora??_Yir?!?P???$
@)a??_Yir?1?P???$
@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??H?0??I?rn???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	BC?+??BC?+??!BC?+??      ??!       "      ??!       *      ??!       2	?
???#4@?
???#4@!?
???#4@:      ??!       B      ??!       J	???M?q?????M?q??!???M?q??R      ??!       Z	???M?q?????M?q??!???M?q??b      ??!       JCPU_ONLYY??H?0??b q?rn???X@