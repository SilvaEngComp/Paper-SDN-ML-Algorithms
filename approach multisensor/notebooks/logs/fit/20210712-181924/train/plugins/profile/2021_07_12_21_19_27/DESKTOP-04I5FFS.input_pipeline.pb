	Ĳ?CR?Q@Ĳ?CR?Q@!Ĳ?CR?Q@	??O?f????O?f??!??O?f??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:Ĳ?CR?Q@)H4???A?a?7?yQ@Y ;7m?i??rEagerKernelExecute 0*	??Q8??@2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Mapjm?k@!?(??YR@)????D?@1?"?`?Q@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?xy:W??!#Z?׼?8@)?Z? m+??1??hdY?7@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?????I??!Hw??j???)????I??1G???????:Preprocessing2U
Iterator::Model::ParallelMapV2\??Ϝ???!~?@u??)\??Ϝ???1~?@u??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????A??!d]?3???)p???????1v?eQ??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateX8I?Ǵ??!??l$??)ץF?g???1FE??)V??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?Y???Ѐ?!?OTɜ#??)?Y???Ѐ?1?OTɜ#??:Preprocessing2F
Iterator::ModelK?ó??!4?.m??)s??/?x??1?!?????:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::PrefetcheQ?E??!%?8?P???)eQ?E??1%?8?P???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?X S??!?K?n^?9@)g??j+?w?1? ??k??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??-??j?!?o?Zp>??)??-??j?1?o?Zp>??:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range??	??`?!?????R??)??	??`?1?????R??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?	????!???Gf??)܂????Y?1?D?U??:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor/PR`L9?!w??lcȉ?)/PR`L9?1w??lcȉ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??O?f??I?Z{&?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	)H4???)H4???!)H4???      ??!       "      ??!       *      ??!       2	?a?7?yQ@?a?7?yQ@!?a?7?yQ@:      ??!       B      ??!       J	 ;7m?i?? ;7m?i??! ;7m?i??R      ??!       Z	 ;7m?i?? ;7m?i??! ;7m?i??b      ??!       JCPU_ONLYY??O?f??b q?Z{&?X@