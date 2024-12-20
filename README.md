# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py


<details>
<summary>Sentiment Classification Logs (70% validation accuracy reached at epoch 83</summary>

```
Epoch 1, loss 31.314978597501195, train accuracy: 49.11%
Validation accuracy: 52.00%
Best Valid accuracy: 52.00%
Epoch 2, loss 31.205760377866955, train accuracy: 51.33%
Validation accuracy: 55.00%
Best Valid accuracy: 55.00%
Epoch 3, loss 31.099179938694686, train accuracy: 53.11%
Validation accuracy: 60.00%
Best Valid accuracy: 60.00%
Epoch 4, loss 30.994554781377676, train accuracy: 55.11%
Validation accuracy: 55.00%
Best Valid accuracy: 60.00%
Epoch 5, loss 30.8921214193426, train accuracy: 57.11%
Validation accuracy: 55.00%
Best Valid accuracy: 60.00%
Epoch 6, loss 30.79181854624879, train accuracy: 58.89%
Validation accuracy: 54.00%
Best Valid accuracy: 60.00%
Epoch 7, loss 30.693573929431462, train accuracy: 59.56%
Validation accuracy: 53.00%
Best Valid accuracy: 60.00%
Epoch 8, loss 30.597317599683766, train accuracy: 60.44%
Validation accuracy: 60.00%
Best Valid accuracy: 60.00%
Epoch 9, loss 30.502982335162248, train accuracy: 61.33%
Validation accuracy: 59.00%
Best Valid accuracy: 60.00%
Epoch 10, loss 30.410503585760498, train accuracy: 62.22%
Validation accuracy: 57.00%
Best Valid accuracy: 60.00%
Epoch 11, loss 30.31981937446124, train accuracy: 63.56%
Validation accuracy: 61.00%
Best Valid accuracy: 61.00%
Epoch 12, loss 30.23087019953687, train accuracy: 64.44%
Validation accuracy: 47.00%
Best Valid accuracy: 61.00%
Epoch 13, loss 30.143598938847873, train accuracy: 65.33%
Validation accuracy: 56.00%
Best Valid accuracy: 61.00%
Epoch 14, loss 30.0579507564566, train accuracy: 66.67%
Validation accuracy: 56.00%
Best Valid accuracy: 61.00%
Epoch 15, loss 29.97387301169486, train accuracy: 67.11%
Validation accuracy: 59.00%
Best Valid accuracy: 61.00%
Epoch 16, loss 29.891315170791493, train accuracy: 67.78%
Validation accuracy: 53.00%
Best Valid accuracy: 61.00%
Epoch 17, loss 29.810228721137772, train accuracy: 67.78%
Validation accuracy: 53.00%
Best Valid accuracy: 61.00%
Epoch 18, loss 29.730567088244822, train accuracy: 68.00%
Validation accuracy: 57.00%
Best Valid accuracy: 61.00%
Epoch 19, loss 29.652285555425756, train accuracy: 68.22%
Validation accuracy: 59.00%
Best Valid accuracy: 61.00%
Epoch 20, loss 29.575341186216185, train accuracy: 68.44%
Validation accuracy: 53.00%
Best Valid accuracy: 61.00%
Epoch 21, loss 29.49969274953132, train accuracy: 68.44%
Validation accuracy: 51.00%
Best Valid accuracy: 61.00%
Epoch 22, loss 29.42530064754306, train accuracy: 68.67%
Validation accuracy: 54.00%
Best Valid accuracy: 61.00%
Epoch 23, loss 29.35212684624947, train accuracy: 69.33%
Validation accuracy: 60.00%
Best Valid accuracy: 61.00%
Epoch 24, loss 29.28013480869795, train accuracy: 69.33%
Validation accuracy: 58.00%
Best Valid accuracy: 61.00%
Epoch 25, loss 29.209289430815936, train accuracy: 70.00%
Validation accuracy: 59.00%
Best Valid accuracy: 61.00%
Epoch 26, loss 29.13955697979476, train accuracy: 69.78%
Validation accuracy: 58.00%
Best Valid accuracy: 61.00%
Epoch 27, loss 29.070905034967954, train accuracy: 70.22%
Validation accuracy: 55.00%
Best Valid accuracy: 61.00%
Epoch 28, loss 29.00330243111923, train accuracy: 70.44%
Validation accuracy: 58.00%
Best Valid accuracy: 61.00%
Epoch 29, loss 28.936719204153103, train accuracy: 70.22%
Validation accuracy: 53.00%
Best Valid accuracy: 61.00%
Epoch 30, loss 28.871126539057357, train accuracy: 70.00%
Validation accuracy: 51.00%
Best Valid accuracy: 61.00%
Epoch 31, loss 28.806496720085434, train accuracy: 70.00%
Validation accuracy: 68.00%
Best Valid accuracy: 68.00%
Epoch 32, loss 28.74280308308519, train accuracy: 70.00%
Validation accuracy: 62.00%
Best Valid accuracy: 68.00%
Epoch 33, loss 28.68001996989979, train accuracy: 70.22%
Validation accuracy: 61.00%
Best Valid accuracy: 68.00%
Epoch 34, loss 28.618122684766764, train accuracy: 70.44%
Validation accuracy: 62.00%
Best Valid accuracy: 68.00%
Epoch 35, loss 28.557087452640992, train accuracy: 70.44%
Validation accuracy: 60.00%
Best Valid accuracy: 68.00%
Epoch 36, loss 28.496891379368552, train accuracy: 70.44%
Validation accuracy: 66.00%
Best Valid accuracy: 68.00%
Epoch 37, loss 28.437512413638956, train accuracy: 70.67%
Validation accuracy: 61.00%
Best Valid accuracy: 68.00%
Epoch 38, loss 28.378929310644832, train accuracy: 70.67%
Validation accuracy: 54.00%
Best Valid accuracy: 68.00%
Epoch 39, loss 28.3211215973794, train accuracy: 70.89%
Validation accuracy: 60.00%
Best Valid accuracy: 68.00%
Epoch 40, loss 28.264069539503552, train accuracy: 70.89%
Validation accuracy: 64.00%
Best Valid accuracy: 68.00%
Epoch 41, loss 28.20775410971663, train accuracy: 70.89%
Validation accuracy: 69.00%
Best Valid accuracy: 69.00%
Epoch 42, loss 28.15215695756618, train accuracy: 70.67%
Validation accuracy: 58.00%
Best Valid accuracy: 69.00%
Epoch 43, loss 28.097260380634374, train accuracy: 70.89%
Validation accuracy: 63.00%
Best Valid accuracy: 69.00%
Epoch 44, loss 28.043047297040758, train accuracy: 71.11%
Validation accuracy: 62.00%
Best Valid accuracy: 69.00%
Epoch 45, loss 27.98950121920269, train accuracy: 71.33%
Validation accuracy: 63.00%
Best Valid accuracy: 69.00%
Epoch 46, loss 27.936606228797334, train accuracy: 71.78%
Validation accuracy: 59.00%
Best Valid accuracy: 69.00%
Epoch 47, loss 27.88434695287086, train accuracy: 71.78%
Validation accuracy: 53.00%
Best Valid accuracy: 69.00%
Epoch 48, loss 27.832708541042443, train accuracy: 71.78%
Validation accuracy: 56.00%
Best Valid accuracy: 69.00%
Epoch 49, loss 27.781676643753226, train accuracy: 71.78%
Validation accuracy: 60.00%
Best Valid accuracy: 69.00%
Epoch 50, loss 27.73123739151168, train accuracy: 72.00%
Validation accuracy: 62.00%
Best Valid accuracy: 69.00%
Epoch 51, loss 27.6813773750893, train accuracy: 72.00%
Validation accuracy: 59.00%
Best Valid accuracy: 69.00%
Epoch 52, loss 27.632083626622467, train accuracy: 72.00%
Validation accuracy: 57.00%
Best Valid accuracy: 69.00%
Epoch 53, loss 27.58334360157781, train accuracy: 72.00%
Validation accuracy: 57.00%
Best Valid accuracy: 69.00%
Epoch 54, loss 27.53514516154065, train accuracy: 72.00%
Validation accuracy: 60.00%
Best Valid accuracy: 69.00%
Epoch 55, loss 27.4874765577877, train accuracy: 72.22%
Validation accuracy: 61.00%
Best Valid accuracy: 69.00%
Epoch 56, loss 27.44032641560694, train accuracy: 72.44%
Validation accuracy: 54.00%
Best Valid accuracy: 69.00%
Epoch 57, loss 27.393683719329267, train accuracy: 72.44%
Validation accuracy: 61.00%
Best Valid accuracy: 69.00%
Epoch 58, loss 27.347537798038086, train accuracy: 72.67%
Validation accuracy: 60.00%
Best Valid accuracy: 69.00%
Epoch 59, loss 27.301878311924614, train accuracy: 72.67%
Validation accuracy: 56.00%
Best Valid accuracy: 69.00%
Epoch 60, loss 27.256695239258242, train accuracy: 72.89%
Validation accuracy: 59.00%
Best Valid accuracy: 69.00%
Epoch 61, loss 27.211978863942416, train accuracy: 72.89%
Validation accuracy: 57.00%
Best Valid accuracy: 69.00%
Epoch 62, loss 27.167719763628384, train accuracy: 72.89%
Validation accuracy: 60.00%
Best Valid accuracy: 69.00%
Epoch 63, loss 27.123908798359906, train accuracy: 73.11%
Validation accuracy: 59.00%
Best Valid accuracy: 69.00%
Epoch 64, loss 27.080537099723834, train accuracy: 73.11%
Validation accuracy: 65.00%
Best Valid accuracy: 69.00%
Epoch 65, loss 27.037596060482183, train accuracy: 72.89%
Validation accuracy: 63.00%
Best Valid accuracy: 69.00%
Epoch 66, loss 26.99507732466303, train accuracy: 72.89%
Validation accuracy: 56.00%
Best Valid accuracy: 69.00%
Epoch 67, loss 26.95297277808796, train accuracy: 72.89%
Validation accuracy: 57.00%
Best Valid accuracy: 69.00%
Epoch 68, loss 26.911274539315677, train accuracy: 72.89%
Validation accuracy: 59.00%
Best Valid accuracy: 69.00%
Epoch 69, loss 26.86997495098156, train accuracy: 72.89%
Validation accuracy: 60.00%
Best Valid accuracy: 69.00%
Epoch 70, loss 26.829066571514907, train accuracy: 72.89%
Validation accuracy: 61.00%
Best Valid accuracy: 69.00%
Epoch 71, loss 26.788542167215137, train accuracy: 73.33%
Validation accuracy: 52.00%
Best Valid accuracy: 69.00%
Epoch 72, loss 26.74839470467076, train accuracy: 73.33%
Validation accuracy: 56.00%
Best Valid accuracy: 69.00%
Epoch 73, loss 26.70861734350436, train accuracy: 73.33%
Validation accuracy: 65.00%
Best Valid accuracy: 69.00%
Epoch 74, loss 26.669203429428347, train accuracy: 73.33%
Validation accuracy: 57.00%
Best Valid accuracy: 69.00%
Epoch 75, loss 26.630146487596893, train accuracy: 73.33%
Validation accuracy: 60.00%
Best Valid accuracy: 69.00%
Epoch 76, loss 26.59144021624, train accuracy: 73.33%
Validation accuracy: 54.00%
Best Valid accuracy: 69.00%
Epoch 77, loss 26.553078480566604, train accuracy: 73.56%
Validation accuracy: 54.00%
Best Valid accuracy: 69.00%
Epoch 78, loss 26.515055306923962, train accuracy: 73.56%
Validation accuracy: 57.00%
Best Valid accuracy: 69.00%
Epoch 79, loss 26.477364877201424, train accuracy: 73.56%
Validation accuracy: 62.00%
Best Valid accuracy: 69.00%
Epoch 80, loss 26.440001523467338, train accuracy: 73.56%
Validation accuracy: 60.00%
Best Valid accuracy: 69.00%
Epoch 81, loss 26.402959722827983, train accuracy: 73.33%
Validation accuracy: 61.00%
Best Valid accuracy: 69.00%
Epoch 82, loss 26.366234092498424, train accuracy: 73.33%
Validation accuracy: 60.00%
Best Valid accuracy: 69.00%
Epoch 83, loss 26.329819385075666, train accuracy: 73.33%
Validation accuracy: 70.00%
Best Valid accuracy: 70.00%
```
</details>




<details>
<summary>MNIST Classification Logs</summary>

```
Epoch 1 loss 2.3014189406409997 valid acc 3/16
Epoch 1 loss 11.512460010463295 valid acc 1/16
Epoch 1 loss 11.496346507012237 valid acc 1/16
Epoch 1 loss 11.55733812731299 valid acc 1/16
Epoch 1 loss 11.50556188893666 valid acc 1/16
Epoch 1 loss 11.447903535395936 valid acc 1/16
Epoch 1 loss 11.485808630111409 valid acc 1/16
Epoch 1 loss 11.4772855490735 valid acc 1/16
Epoch 1 loss 11.42943767121504 valid acc 1/16
Epoch 1 loss 11.513212317530158 valid acc 2/16
Epoch 1 loss 11.4278788887728 valid acc 2/16
Epoch 1 loss 11.429417013290951 valid acc 1/16
Epoch 1 loss 11.400575909755457 valid acc 2/16
Epoch 1 loss 11.418795311046576 valid acc 2/16
Epoch 1 loss 11.407356403116975 valid acc 2/16
Epoch 1 loss 11.431275987990873 valid acc 2/16
Epoch 1 loss 11.363395853939487 valid acc 2/16
Epoch 1 loss 11.359850918159283 valid acc 2/16
Epoch 1 loss 11.376438906507719 valid acc 2/16
Epoch 1 loss 11.4128288926954 valid acc 3/16
Epoch 1 loss 11.408860325877232 valid acc 2/16
Epoch 1 loss 11.333633244609143 valid acc 2/16
Epoch 1 loss 11.309302603157112 valid acc 2/16
Epoch 1 loss 11.365609775754116 valid acc 2/16
Epoch 1 loss 11.346587834860102 valid acc 3/16
Epoch 1 loss 11.358890197995395 valid acc 3/16
Epoch 1 loss 11.34973325002216 valid acc 3/16
Epoch 1 loss 11.203193802260763 valid acc 4/16
Epoch 1 loss 11.156677354169535 valid acc 5/16
Epoch 1 loss 11.234707425032964 valid acc 6/16
Epoch 1 loss 10.979462918794678 valid acc 7/16
Epoch 1 loss 10.909559470184782 valid acc 6/16
Epoch 1 loss 10.737017961230585 valid acc 8/16
Epoch 1 loss 10.59389036615221 valid acc 6/16
Epoch 1 loss 10.41511387439418 valid acc 7/16
Epoch 1 loss 9.929527358932916 valid acc 8/16
Epoch 1 loss 9.450874875958661 valid acc 7/16
Epoch 1 loss 9.25383346930937 valid acc 12/16
Epoch 1 loss 8.087835688404777 valid acc 12/16
Epoch 1 loss 7.895932832564441 valid acc 10/16
Epoch 1 loss 7.006049213298191 valid acc 12/16
Epoch 1 loss 5.658532039180135 valid acc 10/16
Epoch 1 loss 6.982922193304245 valid acc 10/16
Epoch 1 loss 5.32182060507742 valid acc 11/16
Epoch 1 loss 5.961894683938757 valid acc 13/16
Epoch 1 loss 4.596691189844307 valid acc 12/16
Epoch 1 loss 5.9536986126126425 valid acc 12/16
Epoch 1 loss 4.8117538818483165 valid acc 13/16
Epoch 1 loss 4.752935212893377 valid acc 14/16
Epoch 1 loss 5.109047159274094 valid acc 14/16
Epoch 1 loss 5.570316054476939 valid acc 12/16
Epoch 1 loss 4.912505994846358 valid acc 14/16
Epoch 1 loss 4.802145594657262 valid acc 13/16
Epoch 1 loss 3.2314923320282216 valid acc 13/16
Epoch 1 loss 4.456719519842254 valid acc 13/16
Epoch 1 loss 4.4592112803089465 valid acc 14/16
Epoch 1 loss 4.557991279629937 valid acc 13/16
Epoch 1 loss 3.847073203231832 valid acc 14/16
Epoch 1 loss 4.139584010496124 valid acc 14/16
Epoch 1 loss 3.763034357184151 valid acc 14/16
Epoch 1 loss 3.9727636303189957 valid acc 14/16
Epoch 1 loss 4.565285614993844 valid acc 13/16
Epoch 1 loss 4.538561884266262 valid acc 13/16
Epoch 2 loss 0.5117724311217869 valid acc 14/16
Epoch 2 loss 4.7435733020329485 valid acc 15/16
Epoch 2 loss 3.820143470291418 valid acc 14/16
Epoch 2 loss 3.6118791364365315 valid acc 14/16
Epoch 2 loss 3.0617904878203746 valid acc 15/16
Epoch 2 loss 3.01374207796487 valid acc 12/16
Epoch 2 loss 3.3795541340004043 valid acc 14/16
Epoch 2 loss 4.231546919675895 valid acc 14/16
Epoch 2 loss 5.019830187035674 valid acc 14/16
Epoch 2 loss 3.0766256586655576 valid acc 13/16
Epoch 2 loss 3.709656618052784 valid acc 15/16
Epoch 2 loss 4.785684785899814 valid acc 13/16
Epoch 2 loss 4.611512751056978 valid acc 13/16
Epoch 2 loss 4.291835200574952 valid acc 12/16
Epoch 2 loss 6.215652467506734 valid acc 12/16
Epoch 2 loss 3.8204763179150776 valid acc 14/16
Epoch 2 loss 4.496285902024105 valid acc 15/16
Epoch 2 loss 4.325925811681612 valid acc 14/16
Epoch 2 loss 3.4286172427783157 valid acc 15/16
Epoch 2 loss 2.68046983660542 valid acc 15/16
Epoch 2 loss 3.5190410794334666 valid acc 13/16
Epoch 2 loss 2.863252699109471 valid acc 14/16
Epoch 2 loss 1.8476382117427694 valid acc 12/16
Epoch 2 loss 3.3177969791587403 valid acc 14/16
Epoch 2 loss 2.5444763124477685 valid acc 14/16
Epoch 2 loss 2.5856841191317192 valid acc 14/16
Epoch 2 loss 3.990265284739138 valid acc 14/16
Epoch 2 loss 2.9874693821077756 valid acc 14/16
Epoch 2 loss 3.526683800363843 valid acc 13/16
Epoch 2 loss 2.4129363950351808 valid acc 14/16
Epoch 2 loss 4.1019787865533095 valid acc 14/16
Epoch 2 loss 3.138904264979873 valid acc 12/16
Epoch 2 loss 2.2327724221543295 valid acc 15/16
Epoch 2 loss 3.17120115586143 valid acc 15/16
Epoch 2 loss 6.042216831432823 valid acc 14/16
Epoch 2 loss 2.60166990601077 valid acc 13/16
Epoch 2 loss 2.293597004880752 valid acc 14/16
Epoch 2 loss 3.149600870481062 valid acc 14/16
Epoch 2 loss 2.780323743425179 valid acc 14/16
Epoch 2 loss 3.332318909268554 valid acc 14/16
Epoch 2 loss 2.719190261028012 valid acc 15/16
Epoch 2 loss 3.4528798976054125 valid acc 15/16
Epoch 2 loss 3.0489208131059358 valid acc 14/16
Epoch 2 loss 1.950490604045235 valid acc 15/16
Epoch 2 loss 3.1263385202032663 valid acc 13/16
Epoch 2 loss 1.622442337178901 valid acc 14/16
Epoch 2 loss 2.985981687098856 valid acc 14/16
Epoch 2 loss 2.9886774680796373 valid acc 15/16
Epoch 2 loss 2.7232486878384865 valid acc 14/16
Epoch 2 loss 1.844038708698893 valid acc 15/16
Epoch 2 loss 3.059321608218379 valid acc 14/16
Epoch 2 loss 2.729426040656595 valid acc 14/16
Epoch 2 loss 3.128640145471026 valid acc 15/16
Epoch 2 loss 1.8918895353334921 valid acc 14/16
Epoch 2 loss 3.4321899930893554 valid acc 14/16
Epoch 2 loss 2.720112066178734 valid acc 15/16
Epoch 2 loss 2.5314104526819943 valid acc 14/16
Epoch 2 loss 2.853808291509427 valid acc 15/16
Epoch 2 loss 2.81064291616883 valid acc 14/16
Epoch 2 loss 3.2096509337759236 valid acc 14/16
Epoch 2 loss 2.171535147679296 valid acc 15/16
Epoch 2 loss 2.389205258263456 valid acc 14/16
Epoch 2 loss 2.947902224483769 valid acc 15/16
Epoch 3 loss 0.204700528666711 valid acc 15/16
Epoch 3 loss 2.6452689016601387 valid acc 13/16
Epoch 3 loss 2.655626775327136 valid acc 15/16
Epoch 3 loss 2.799593646761219 valid acc 14/16
Epoch 3 loss 2.4381529774810584 valid acc 14/16
Epoch 3 loss 1.9959233264703875 valid acc 13/16
Epoch 3 loss 2.5540020725292956 valid acc 15/16
Epoch 3 loss 3.4824255259735004 valid acc 15/16
Epoch 3 loss 3.124644095787662 valid acc 15/16
Epoch 3 loss 2.535525777712589 valid acc 15/16
Epoch 3 loss 3.3789108273297144 valid acc 15/16
Epoch 3 loss 3.829893302056814 valid acc 14/16
Epoch 3 loss 3.3305443638324044 valid acc 15/16
Epoch 3 loss 3.538250671810104 valid acc 13/16
Epoch 3 loss 4.402614837468697 valid acc 13/16
Epoch 3 loss 2.521013151234107 valid acc 15/16
Epoch 3 loss 3.479933206122576 valid acc 15/16
Epoch 3 loss 3.5508582681466576 valid acc 15/16
Epoch 3 loss 2.269071536786634 valid acc 14/16
Epoch 3 loss 1.9323975648132719 valid acc 14/16
Epoch 3 loss 2.853717297350405 valid acc 13/16
Epoch 3 loss 2.535746213509556 valid acc 13/16
Epoch 3 loss 1.545373492650804 valid acc 13/16
Epoch 3 loss 2.871180634758475 valid acc 14/16
Epoch 3 loss 2.0155459008584806 valid acc 14/16
Epoch 3 loss 3.4527728345609465 valid acc 14/16
Epoch 3 loss 2.7815866565664615 valid acc 14/16
Epoch 3 loss 2.6996085894812567 valid acc 14/16
Epoch 3 loss 1.7577150327161695 valid acc 15/16
Epoch 3 loss 1.8093746370437571 valid acc 15/16
Epoch 3 loss 2.6322782368486846 valid acc 15/16
Epoch 3 loss 2.493656380827929 valid acc 13/16
Epoch 3 loss 1.5068068216907524 valid acc 14/16
Epoch 3 loss 2.1443664276793686 valid acc 15/16
Epoch 3 loss 3.5255325033701364 valid acc 14/16
Epoch 3 loss 1.9649831639805833 valid acc 14/16
Epoch 3 loss 2.1617181830015144 valid acc 15/16
Epoch 3 loss 2.8330136022330503 valid acc 15/16
Epoch 3 loss 2.0526151288592107 valid acc 14/16
Epoch 3 loss 2.22762481763259 valid acc 15/16
Epoch 3 loss 1.5311986762313123 valid acc 15/16
Epoch 3 loss 2.8210405317412013 valid acc 15/16
Epoch 3 loss 1.9360389742003878 valid acc 15/16
Epoch 3 loss 1.8089812086156416 valid acc 15/16
Epoch 3 loss 2.4598993160353704 valid acc 15/16
Epoch 3 loss 1.6889362102591254 valid acc 14/16
Epoch 3 loss 2.1169078882817414 valid acc 15/16
Epoch 3 loss 2.7861314396906427 valid acc 15/16
Epoch 3 loss 2.189649050507022 valid acc 15/16
Epoch 3 loss 1.7965292099819836 valid acc 15/16
Epoch 3 loss 1.983884513598066 valid acc 15/16
Epoch 3 loss 2.013116161585693 valid acc 15/16
Epoch 3 loss 2.6251222213518677 valid acc 15/16
Epoch 3 loss 1.6727728039853584 valid acc 15/16
Epoch 3 loss 2.9253290769621643 valid acc 14/16
Epoch 3 loss 1.961219538926568 valid acc 15/16
Epoch 3 loss 2.4942703139926112 valid acc 15/16
Epoch 3 loss 2.200494440535074 valid acc 15/16
Epoch 3 loss 2.1647000518173334 valid acc 15/16
Epoch 3 loss 2.286437750579506 valid acc 14/16
Epoch 3 loss 1.9711389637399592 valid acc 15/16
Epoch 3 loss 1.7596775861936804 valid acc 15/16
Epoch 3 loss 2.5456186334198447 valid acc 15/16
Epoch 4 loss 0.3218323101250175 valid acc 15/16
Epoch 4 loss 2.0263593515537686 valid acc 13/16
Epoch 4 loss 1.7606036745687415 valid acc 15/16
Epoch 4 loss 1.8941393715409056 valid acc 14/16
Epoch 4 loss 1.4362014427462018 valid acc 14/16
Epoch 4 loss 2.0500260371196615 valid acc 14/16
Epoch 4 loss 2.067469165165492 valid acc 15/16
Epoch 4 loss 2.9489275439406795 valid acc 15/16
Epoch 4 loss 3.542189717946012 valid acc 15/16
Epoch 4 loss 2.4807704851692947 valid acc 13/16
Epoch 4 loss 2.3017335405335815 valid acc 15/16
Epoch 4 loss 4.179884899092453 valid acc 15/16
Epoch 4 loss 2.4715881708126535 valid acc 14/16
Epoch 4 loss 3.2145067329394013 valid acc 13/16
Epoch 4 loss 4.446430162810039 valid acc 13/16
Epoch 4 loss 2.653949463423581 valid acc 15/16
Epoch 4 loss 2.991489676720786 valid acc 15/16
Epoch 4 loss 3.543523406026012 valid acc 13/16
Epoch 4 loss 2.321821801146898 valid acc 14/16
Epoch 4 loss 2.142710503620358 valid acc 15/16
Epoch 4 loss 2.3815757777301556 valid acc 14/16
Epoch 4 loss 1.9039574479811745 valid acc 15/16
Epoch 4 loss 1.4587363968647495 valid acc 13/16
Epoch 4 loss 2.5933542626856436 valid acc 15/16
Epoch 4 loss 2.045112937396913 valid acc 15/16
Epoch 4 loss 2.027645589732311 valid acc 15/16
Epoch 4 loss 2.260053149402992 valid acc 14/16
Epoch 4 loss 1.505152592540295 valid acc 15/16
Epoch 4 loss 1.3116487674773405 valid acc 15/16
Epoch 4 loss 0.9557076432742266 valid acc 15/16
Epoch 4 loss 2.1281486262642417 valid acc 15/16
Epoch 4 loss 1.7046927149639446 valid acc 15/16
Epoch 4 loss 0.8841971862637326 valid acc 15/16
Epoch 4 loss 1.968049094528144 valid acc 15/16
Epoch 4 loss 3.9166437233142624 valid acc 14/16
Epoch 4 loss 2.372496680783792 valid acc 14/16
Epoch 4 loss 2.182936979198618 valid acc 14/16
Epoch 4 loss 2.043494520142622 valid acc 14/16
Epoch 4 loss 1.9638619017682593 valid acc 14/16
Epoch 4 loss 1.6154732849747684 valid acc 15/16
Epoch 4 loss 1.5831317789319808 valid acc 14/16
Epoch 4 loss 3.087122957037838 valid acc 14/16
Epoch 4 loss 1.7356675240785804 valid acc 15/16
Epoch 4 loss 1.3274298504679067 valid acc 15/16
Epoch 4 loss 1.9392656917954596 valid acc 15/16
Epoch 4 loss 1.1330255485255203 valid acc 15/16
Epoch 4 loss 1.7832353868894053 valid acc 15/16
Epoch 4 loss 1.8816158854653167 valid acc 14/16
Epoch 4 loss 1.3935404100424855 valid acc 15/16
Epoch 4 loss 1.3720754083365418 valid acc 15/16
Epoch 4 loss 1.959144929770865 valid acc 15/16
Epoch 4 loss 2.0992197686711513 valid acc 15/16
Epoch 4 loss 2.37072640967951 valid acc 15/16
Epoch 4 loss 1.7900746276357533 valid acc 14/16
Epoch 4 loss 2.7305988432258577 valid acc 14/16
Epoch 4 loss 1.948485795611344 valid acc 14/16
Epoch 4 loss 1.4447196797542334 valid acc 14/16
Epoch 4 loss 1.874135932627901 valid acc 14/16
Epoch 4 loss 2.1623814697846564 valid acc 15/16
Epoch 4 loss 2.3749785918799757 valid acc 15/16
Epoch 4 loss 1.6938256119054014 valid acc 15/16
Epoch 4 loss 1.8015918112004685 valid acc 15/16
Epoch 4 loss 2.557956585428252 valid acc 14/16
Epoch 5 loss 0.2591547666646195 valid acc 15/16
Epoch 5 loss 2.126862683091962 valid acc 15/16
Epoch 5 loss 1.7768659399064726 valid acc 14/16
Epoch 5 loss 1.9209856197604025 valid acc 13/16
Epoch 5 loss 1.5031108403598463 valid acc 14/16
Epoch 5 loss 1.5313649093835442 valid acc 15/16
Epoch 5 loss 1.7816061298953985 valid acc 14/16
Epoch 5 loss 2.711912913870491 valid acc 13/16
Epoch 5 loss 2.0977062078352455 valid acc 15/16
Epoch 5 loss 1.4525314429220786 valid acc 15/16
Epoch 5 loss 1.2662266176708887 valid acc 15/16
Epoch 5 loss 3.223579888513308 valid acc 13/16
Epoch 5 loss 2.704825592293406 valid acc 14/16
Epoch 5 loss 2.339401003609219 valid acc 15/16
Epoch 5 loss 3.4410524936118603 valid acc 13/16
Epoch 5 loss 1.8616100752597573 valid acc 15/16
Epoch 5 loss 3.1555086093384137 valid acc 15/16
Epoch 5 loss 2.6934110919039416 valid acc 14/16
Epoch 5 loss 2.475527583240746 valid acc 16/16
Epoch 5 loss 2.4174080126040067 valid acc 15/16
Epoch 5 loss 1.9312685456583971 valid acc 14/16
Epoch 5 loss 2.4542011362819927 valid acc 16/16
Epoch 5 loss 1.3609376794769497 valid acc 14/16
Epoch 5 loss 3.0753639235497956 valid acc 15/16
Epoch 5 loss 1.622646130199562 valid acc 15/16
Epoch 5 loss 1.8352757126723986 valid acc 16/16
Epoch 5 loss 2.3924788797947594 valid acc 15/16
Epoch 5 loss 1.6397314408399497 valid acc 15/16
Epoch 5 loss 1.3736805815771513 valid acc 14/16
Epoch 5 loss 1.5791618756949268 valid acc 14/16
Epoch 5 loss 2.369289459567857 valid acc 14/16
Epoch 5 loss 2.402893813809208 valid acc 14/16
Epoch 5 loss 1.1944888930705653 valid acc 14/16
Epoch 5 loss 1.858023429081408 valid acc 15/16
Epoch 5 loss 3.041078032672422 valid acc 15/16
Epoch 5 loss 2.3012798309471454 valid acc 14/16
Epoch 5 loss 2.3296394762295893 valid acc 14/16
Epoch 5 loss 2.592558939465545 valid acc 16/16
Epoch 5 loss 2.2775100646882365 valid acc 15/16
Epoch 5 loss 2.022277755102724 valid acc 16/16
Epoch 5 loss 1.9032747347404135 valid acc 15/16
Epoch 5 loss 1.9144911763026664 valid acc 16/16
Epoch 5 loss 2.0067128950641226 valid acc 13/16
Epoch 5 loss 1.5735456030679718 valid acc 15/16
Epoch 5 loss 4.235163296739432 valid acc 15/16
Epoch 5 loss 1.4945326657673257 valid acc 14/16
Epoch 5 loss 1.8686283865934818 valid acc 16/16
Epoch 5 loss 2.4025791582884235 valid acc 13/16
Epoch 5 loss 2.0141922338564204 valid acc 15/16
Epoch 5 loss 1.6370305092636683 valid acc 16/16
Epoch 5 loss 2.3826645557594492 valid acc 16/16
Epoch 5 loss 2.066230610234962 valid acc 16/16
Epoch 5 loss 2.3811898915757093 valid acc 16/16
Epoch 5 loss 1.660172666914372 valid acc 16/16
Epoch 5 loss 2.226124485943311 valid acc 13/16
Epoch 5 loss 1.7587782169822384 valid acc 16/16
Epoch 5 loss 2.2055491264972296 valid acc 16/16
Epoch 5 loss 1.8779902609067636 valid acc 16/16
Epoch 5 loss 2.111646798568189 valid acc 16/16
Epoch 5 loss 2.2745672585846224 valid acc 15/16
Epoch 5 loss 1.5355662322552694 valid acc 14/16
Epoch 5 loss 2.8271551835296145 valid acc 15/16
Epoch 5 loss 2.4829226552803925 valid acc 13/16
Epoch 6 loss 0.15094724979835963 valid acc 14/16
Epoch 6 loss 2.046340627600712 valid acc 16/16
Epoch 6 loss 1.8892347570394195 valid acc 15/16
Epoch 6 loss 2.355336490968214 valid acc 14/16
Epoch 6 loss 1.328112133906552 valid acc 15/16
Epoch 6 loss 1.8324787957045532 valid acc 15/16
Epoch 6 loss 1.7187053756541184 valid acc 14/16
Epoch 6 loss 2.8057058952248513 valid acc 15/16
Epoch 6 loss 1.8881733644126724 valid acc 13/16
Epoch 6 loss 1.7703365467579986 valid acc 16/16
Epoch 6 loss 2.138028754153103 valid acc 15/16
Epoch 6 loss 5.0545141794517345 valid acc 15/16
Epoch 6 loss 2.9109687647383558 valid acc 16/16
Epoch 6 loss 2.811548978005413 valid acc 14/16
Epoch 6 loss 3.9578039733351678 valid acc 14/16
Epoch 6 loss 2.5711094830083554 valid acc 14/16
Epoch 6 loss 3.3588852864322183 valid acc 15/16
Epoch 6 loss 2.6901179161293047 valid acc 15/16
Epoch 6 loss 2.0231439286786648 valid acc 16/16
Epoch 6 loss 1.1753378620696764 valid acc 16/16
Epoch 6 loss 1.8905984040053445 valid acc 15/16
Epoch 6 loss 1.5193432298405618 valid acc 15/16
Epoch 6 loss 0.8280523028036996 valid acc 15/16
Epoch 6 loss 1.9142997160054982 valid acc 15/16
Epoch 6 loss 1.173370123912709 valid acc 15/16
Epoch 6 loss 2.128860332532025 valid acc 15/16
Epoch 6 loss 2.6436809333948688 valid acc 15/16
Epoch 6 loss 2.046459820216239 valid acc 15/16
Epoch 6 loss 2.4514504845874847 valid acc 15/16
Epoch 6 loss 1.7556611995086386 valid acc 15/16
Epoch 6 loss 2.2974113669421166 valid acc 15/16
Epoch 6 loss 3.719862992964475 valid acc 14/16
Epoch 6 loss 1.7184938912804204 valid acc 14/16
Epoch 6 loss 2.047550577571887 valid acc 16/16
Epoch 6 loss 3.8426256237519745 valid acc 16/16
Epoch 6 loss 2.223697384205663 valid acc 16/16
Epoch 6 loss 1.7407079080853833 valid acc 15/16
Epoch 6 loss 2.4740333095134837 valid acc 16/16
Epoch 6 loss 1.9395589419028292 valid acc 16/16
Epoch 6 loss 1.891360055449615 valid acc 16/16
Epoch 6 loss 1.6243897668853688 valid acc 16/16
Epoch 6 loss 1.9612458574851028 valid acc 16/16
Epoch 6 loss 1.354749727738002 valid acc 16/16
Epoch 6 loss 1.5469991347884005 valid acc 16/16
Epoch 6 loss 1.9752844007354096 valid acc 14/16
Epoch 6 loss 1.364948272405266 valid acc 15/16
Epoch 6 loss 2.08430449876222 valid acc 15/16
Epoch 6 loss 2.76689706814246 valid acc 15/16
Epoch 6 loss 1.6644411268339605 valid acc 16/16
Epoch 6 loss 1.8239702703442346 valid acc 16/16
Epoch 6 loss 2.142111112886804 valid acc 16/16
Epoch 6 loss 1.9858655555618294 valid acc 15/16
Epoch 6 loss 2.2610210603138543 valid acc 15/16
Epoch 6 loss 1.9906817480686976 valid acc 16/16
Epoch 6 loss 2.317124881460865 valid acc 14/16
Epoch 6 loss 1.8474375471103004 valid acc 15/16
Epoch 6 loss 1.8562427754926616 valid acc 15/16
Epoch 6 loss 1.7825716443028772 valid acc 15/16
Epoch 6 loss 2.126511861443536 valid acc 15/16
Epoch 6 loss 2.3440741950103856 valid acc 16/16
Epoch 6 loss 1.625057896551308 valid acc 15/16
Epoch 6 loss 2.616708581725492 valid acc 14/16
Epoch 6 loss 2.1442927990109166 valid acc 13/16
Epoch 7 loss 0.061158130475766896 valid acc 14/16
Epoch 7 loss 1.4330961575986592 valid acc 15/16
Epoch 7 loss 1.6967068671963883 valid acc 15/16
Epoch 7 loss 1.673857828368184 valid acc 15/16
Epoch 7 loss 1.3032297629484688 valid acc 15/16
Epoch 7 loss 1.1480959467440366 valid acc 15/16
Epoch 7 loss 1.5861432523279215 valid acc 15/16
Epoch 7 loss 2.417092813340152 valid acc 15/16
Epoch 7 loss 1.9293930156871926 valid acc 16/16
Epoch 7 loss 0.8564970861504074 valid acc 16/16
Epoch 7 loss 1.7310440893152619 valid acc 16/16
Epoch 7 loss 1.8645200637201076 valid acc 14/16
Epoch 7 loss 2.0471477923082912 valid acc 15/16
Epoch 7 loss 2.2192640665343526 valid acc 15/16
Epoch 7 loss 3.062862722153094 valid acc 15/16
Epoch 7 loss 2.3324362899907345 valid acc 15/16
Epoch 7 loss 4.200846281571962 valid acc 14/16
Epoch 7 loss 3.355652580726909 valid acc 14/16
Epoch 7 loss 1.5978788762589744 valid acc 16/16
Epoch 7 loss 1.4524809815845041 valid acc 16/16
Epoch 7 loss 1.9330765787605146 valid acc 15/16
Epoch 7 loss 1.1265753631213964 valid acc 15/16
Epoch 7 loss 0.8613518153597084 valid acc 16/16
Epoch 7 loss 1.6348682330272164 valid acc 15/16
Epoch 7 loss 1.4838761780804162 valid acc 15/16
Epoch 7 loss 1.4039083426650913 valid acc 15/16
Epoch 7 loss 1.4256158709267677 valid acc 15/16
Epoch 7 loss 1.0364439967375048 valid acc 16/16
Epoch 7 loss 1.4450983118685574 valid acc 15/16
Epoch 7 loss 0.8907281876828871 valid acc 15/16
Epoch 7 loss 2.0661918555014576 valid acc 14/16
Epoch 7 loss 3.851587450541886 valid acc 14/16
Epoch 7 loss 1.4024241254936107 valid acc 13/16
Epoch 7 loss 1.8759118921527147 valid acc 15/16
Epoch 7 loss 3.438437938385016 valid acc 16/16
Epoch 7 loss 2.2842417680306397 valid acc 16/16
Epoch 7 loss 2.3757622702287158 valid acc 14/16
Epoch 7 loss 1.490395122712012 valid acc 15/16
Epoch 7 loss 1.5505213398635747 valid acc 16/16
Epoch 7 loss 1.9030421859845337 valid acc 15/16
Epoch 7 loss 1.4695444977625065 valid acc 14/16
Epoch 7 loss 2.2269726608307376 valid acc 16/16
Epoch 7 loss 1.378915563594068 valid acc 16/16
Epoch 7 loss 1.2237568842310187 valid acc 16/16
Epoch 7 loss 1.4083269976436164 valid acc 15/16
Epoch 7 loss 1.2529063749327107 valid acc 16/16
Epoch 7 loss 1.0570584584122387 valid acc 16/16
Epoch 7 loss 1.9119003778779273 valid acc 15/16
Epoch 7 loss 1.2303063037844382 valid acc 16/16
Epoch 7 loss 1.33349093765799 valid acc 15/16
Epoch 7 loss 1.6553294317414184 valid acc 15/16
Epoch 7 loss 2.007608363635292 valid acc 16/16
Epoch 7 loss 2.2326324305817 valid acc 15/16
Epoch 7 loss 1.9658849813868946 valid acc 16/16
Epoch 7 loss 2.4191103299228938 valid acc 14/16
Epoch 7 loss 1.8317785629390413 valid acc 14/16
Epoch 7 loss 3.2819461107248333 valid acc 14/16
Epoch 7 loss 1.4705685135115902 valid acc 14/16
Epoch 7 loss 2.2324161888287755 valid acc 14/16
Epoch 7 loss 2.8800434377782604 valid acc 14/16
Epoch 7 loss 1.9432367843471234 valid acc 16/16
Epoch 7 loss 1.1057858440646298 valid acc 16/16
Epoch 7 loss 2.2702428867858933 valid acc 15/16
Epoch 8 loss 0.401838686718374 valid acc 15/16
Epoch 8 loss 1.848646992106341 valid acc 16/16
Epoch 8 loss 2.047398658160611 valid acc 16/16
Epoch 8 loss 1.4415449389119 valid acc 14/16
Epoch 8 loss 1.406396495021358 valid acc 14/16
Epoch 8 loss 1.6502130354482614 valid acc 14/16
Epoch 8 loss 1.911464165741477 valid acc 13/16
Epoch 8 loss 2.027272510413925 valid acc 13/16
Epoch 8 loss 1.7986645397779668 valid acc 13/16
Epoch 8 loss 1.3784707729241972 valid acc 15/16
Epoch 8 loss 1.7048467848872995 valid acc 15/16
Epoch 8 loss 2.0384024568311823 valid acc 13/16
Epoch 8 loss 2.1226538632002354 valid acc 14/16
Epoch 8 loss 2.255896164615591 valid acc 13/16
Epoch 8 loss 3.0634641324566285 valid acc 15/16
Epoch 8 loss 2.421038050040496 valid acc 15/16
Epoch 8 loss 3.6015268769501905 valid acc 16/16
Epoch 8 loss 1.5227539211592376 valid acc 15/16
Epoch 8 loss 1.4465514689045234 valid acc 16/16
Epoch 8 loss 1.7038590326907752 valid acc 15/16
Epoch 8 loss 1.6405877324042513 valid acc 14/16
Epoch 8 loss 1.0850882527013213 valid acc 16/16
Epoch 8 loss 0.338025314880862 valid acc 16/16
Epoch 8 loss 1.04977360687574 valid acc 15/16
Epoch 8 loss 1.1174573047929373 valid acc 16/16
Epoch 8 loss 1.8240969698867193 valid acc 16/16
Epoch 8 loss 1.9932192231492363 valid acc 16/16
Epoch 8 loss 0.8328397637732496 valid acc 16/16
Epoch 8 loss 0.8616114602653351 valid acc 16/16
Epoch 8 loss 0.6016553847861313 valid acc 16/16
Epoch 8 loss 1.721615118803164 valid acc 15/16
Epoch 8 loss 2.474644920343539 valid acc 15/16
Epoch 8 loss 1.0811779444047338 valid acc 15/16
Epoch 8 loss 1.8297329812800036 valid acc 16/16
Epoch 8 loss 2.4841230391670304 valid acc 16/16
Epoch 8 loss 0.9300214332541158 valid acc 16/16
Epoch 8 loss 1.3513138122045825 valid acc 14/16
Epoch 8 loss 1.1900201418783014 valid acc 16/16
Epoch 8 loss 1.1803840538819717 valid acc 15/16
Epoch 8 loss 1.3202057255765192 valid acc 15/16
Epoch 8 loss 0.9737509507446043 valid acc 13/16
Epoch 8 loss 2.0428259626131875 valid acc 15/16
Epoch 8 loss 1.0711581766857683 valid acc 14/16
Epoch 8 loss 0.9218720875071788 valid acc 15/16
Epoch 8 loss 1.3283562785044225 valid acc 16/16
Epoch 8 loss 0.7182988700993684 valid acc 16/16
Epoch 8 loss 1.47968966789762 valid acc 15/16
Epoch 8 loss 1.9915232053768785 valid acc 15/16
Epoch 8 loss 1.4866064943325523 valid acc 15/16
Epoch 8 loss 1.163270883708359 valid acc 15/16
Epoch 8 loss 1.4728602423868724 valid acc 15/16
Epoch 8 loss 1.8207225120763053 valid acc 16/16
Epoch 8 loss 2.5696732417877737 valid acc 16/16
Epoch 8 loss 0.7441129412191212 valid acc 16/16
Epoch 8 loss 1.457719477951209 valid acc 15/16
Epoch 8 loss 0.8390595119395724 valid acc 15/16
Epoch 8 loss 1.2272467337263162 valid acc 16/16
Epoch 8 loss 1.4776941493372677 valid acc 16/16
Epoch 8 loss 2.7901288704024942 valid acc 14/16
Epoch 8 loss 1.8390402284249114 valid acc 15/16
Epoch 8 loss 1.5238224683505628 valid acc 16/16
Epoch 8 loss 0.9981042717246847 valid acc 15/16
Epoch 8 loss 1.7522511106491656 valid acc 14/16
Epoch 9 loss 0.06240739434035406 valid acc 16/16
Epoch 9 loss 1.4042023177616016 valid acc 15/16
Epoch 9 loss 1.3497973577232274 valid acc 16/16
Epoch 9 loss 1.7561096593738543 valid acc 16/16
Epoch 9 loss 0.592036412199915 valid acc 15/16
Epoch 9 loss 0.9743360872013951 valid acc 15/16
Epoch 9 loss 1.2650932721913932 valid acc 14/16
Epoch 9 loss 1.270424507477816 valid acc 14/16
Epoch 9 loss 1.0432485083560827 valid acc 16/16
Epoch 9 loss 0.6125032990422755 valid acc 16/16
Epoch 9 loss 1.5192249469278154 valid acc 16/16
Epoch 9 loss 1.9956419353962964 valid acc 16/16
Epoch 9 loss 1.7091646265875204 valid acc 15/16
Epoch 9 loss 2.0893274870113734 valid acc 13/16
Epoch 9 loss 2.941799773191512 valid acc 14/16
Epoch 9 loss 1.882980323440446 valid acc 16/16
Epoch 9 loss 2.626176296154036 valid acc 16/16
Epoch 9 loss 2.2777524268952454 valid acc 15/16
Epoch 9 loss 1.82404294244437 valid acc 16/16
Epoch 9 loss 1.2318176939491439 valid acc 16/16
Epoch 9 loss 1.7319902284922295 valid acc 16/16
Epoch 9 loss 0.6545012500843418 valid acc 15/16
Epoch 9 loss 0.7143258369523555 valid acc 15/16
Epoch 9 loss 0.990770925270361 valid acc 15/16
Epoch 9 loss 1.1496962453321062 valid acc 15/16
Epoch 9 loss 1.619446572596423 valid acc 15/16
Epoch 9 loss 1.839696785037788 valid acc 15/16
Epoch 9 loss 0.732509462222815 valid acc 14/16
Epoch 9 loss 0.9049482542295779 valid acc 15/16
Epoch 9 loss 0.5960222372066618 valid acc 15/16
Epoch 9 loss 1.2727681857107471 valid acc 15/16
Epoch 9 loss 2.1229063490379794 valid acc 13/16
Epoch 9 loss 1.0436764923604052 valid acc 14/16
Epoch 9 loss 1.5135591015389496 valid acc 15/16
Epoch 9 loss 3.339046077214629 valid acc 14/16
Epoch 9 loss 1.7382122833741134 valid acc 14/16
Epoch 9 loss 1.8392651449857225 valid acc 15/16
Epoch 9 loss 1.5225375083613895 valid acc 16/16
Epoch 9 loss 1.4735324279575361 valid acc 15/16
Epoch 9 loss 2.15450747932537 valid acc 14/16
Epoch 9 loss 1.437669668650293 valid acc 14/16
Epoch 9 loss 1.5849332256438449 valid acc 14/16
Epoch 9 loss 2.1040595222212493 valid acc 14/16
Epoch 9 loss 0.9521585790486821 valid acc 15/16
Epoch 9 loss 1.7362625582972315 valid acc 15/16
Epoch 9 loss 0.6919760260586731 valid acc 16/16
Epoch 9 loss 1.3261071090004435 valid acc 15/16
Epoch 9 loss 1.6295885980437994 valid acc 16/16
Epoch 9 loss 1.2426571206985721 valid acc 15/16
Epoch 9 loss 1.3103896322761646 valid acc 15/16
Epoch 9 loss 1.291408314490588 valid acc 14/16
Epoch 9 loss 1.4132510754830612 valid acc 14/16
Epoch 9 loss 1.788389587968969 valid acc 15/16
Epoch 9 loss 0.9184651521327152 valid acc 16/16
Epoch 9 loss 1.5016089023935153 valid acc 14/16
Epoch 9 loss 0.9464164788756804 valid acc 14/16
Epoch 9 loss 1.8297314475783044 valid acc 16/16
Epoch 9 loss 0.9279526300093998 valid acc 15/16
Epoch 9 loss 1.4348129642056209 valid acc 16/16
Epoch 9 loss 2.3923726674107098 valid acc 16/16
Epoch 9 loss 1.0738758692379449 valid acc 15/16
Epoch 9 loss 0.7179585822901077 valid acc 16/16
Epoch 9 loss 1.5953709509946252 valid acc 16/16
Epoch 10 loss 0.08840340677360388 valid acc 16/16
Epoch 10 loss 1.1696119832070666 valid acc 16/16
Epoch 10 loss 1.3179894517974793 valid acc 15/16
Epoch 10 loss 0.9125173403773947 valid acc 16/16
Epoch 10 loss 0.6540828861160002 valid acc 15/16
Epoch 10 loss 0.9740241705561629 valid acc 15/16
Epoch 10 loss 1.4182600838111497 valid acc 14/16
Epoch 10 loss 1.245291678921056 valid acc 14/16
Epoch 10 loss 0.9234846625984833 valid acc 15/16
Epoch 10 loss 0.7515848214406857 valid acc 15/16
Epoch 10 loss 0.8218597986073477 valid acc 15/16
Epoch 10 loss 1.2448482052440295 valid acc 16/16
Epoch 10 loss 1.1214970075003738 valid acc 16/16
Epoch 10 loss 1.137324242029996 valid acc 15/16
Epoch 10 loss 2.475467587234707 valid acc 15/16
Epoch 10 loss 1.3310321023918035 valid acc 16/16
Epoch 10 loss 2.4924674820673385 valid acc 16/16
Epoch 10 loss 1.9193438758244987 valid acc 15/16
Epoch 10 loss 1.036546914849038 valid acc 16/16
Epoch 10 loss 0.7562262183220484 valid acc 16/16
Epoch 10 loss 1.4268031335846967 valid acc 15/16
Epoch 10 loss 0.6542263883956887 valid acc 15/16
Epoch 10 loss 0.4359036823867905 valid acc 15/16
Epoch 10 loss 0.648679391137104 valid acc 15/16
Epoch 10 loss 0.6734926534466048 valid acc 14/16
Epoch 10 loss 0.8877668539643832 valid acc 15/16
Epoch 10 loss 1.3584585721665952 valid acc 15/16
Epoch 10 loss 0.5474985989532178 valid acc 15/16
Epoch 10 loss 0.8139031894706401 valid acc 14/16
Epoch 10 loss 0.4704649020106647 valid acc 14/16
Epoch 10 loss 0.6124827747255062 valid acc 14/16
Epoch 10 loss 1.6289036804923511 valid acc 14/16
Epoch 10 loss 0.5935816489828402 valid acc 14/16
Epoch 10 loss 1.0067299053470213 valid acc 16/16
Epoch 10 loss 1.4160530993681761 valid acc 16/16
Epoch 10 loss 1.1311524000751894 valid acc 16/16
Epoch 10 loss 1.079963188560844 valid acc 16/16
Epoch 10 loss 1.2079630467333993 valid acc 16/16
Epoch 10 loss 0.6768869306881216 valid acc 16/16
Epoch 10 loss 0.9044488327820444 valid acc 16/16
Epoch 10 loss 0.43949103989045013 valid acc 16/16
Epoch 10 loss 1.1443797523600425 valid acc 16/16
Epoch 10 loss 0.6116948966687649 valid acc 16/16
Epoch 10 loss 0.5027900372847142 valid acc 16/16
Epoch 10 loss 1.0275091218890922 valid acc 16/16
Epoch 10 loss 0.4623600968693218 valid acc 16/16
Epoch 10 loss 1.0167648929210387 valid acc 16/16
Epoch 10 loss 1.189902932220526 valid acc 16/16
Epoch 10 loss 0.7639393989989065 valid acc 16/16
Epoch 10 loss 1.0342196688019778 valid acc 16/16
Epoch 10 loss 0.9003669453098444 valid acc 15/16
Epoch 10 loss 0.9419329800884687 valid acc 16/16
Epoch 10 loss 1.4866119034423826 valid acc 15/16
Epoch 10 loss 0.7943965877899808 valid acc 16/16
Epoch 10 loss 1.4259982572733718 valid acc 15/16
Epoch 10 loss 0.9617180101776623 valid acc 16/16
Epoch 10 loss 0.835033023555716 valid acc 15/16
Epoch 10 loss 0.8969051398314654 valid acc 16/16
Epoch 10 loss 1.2922849688033955 valid acc 16/16
Epoch 10 loss 1.2114899340856613 valid acc 16/16
Epoch 10 loss 0.8663067500576258 valid acc 16/16
Epoch 10 loss 0.6083509287102168 valid acc 16/16
Epoch 10 loss 1.753822137706343 valid acc 16/16
Epoch 11 loss 0.015619756312736887 valid acc 16/16
Epoch 11 loss 1.1184311747640634 valid acc 16/16
Epoch 11 loss 1.0855425512716579 valid acc 15/16
Epoch 11 loss 1.3687763436528886 valid acc 15/16
Epoch 11 loss 0.49182920992588236 valid acc 15/16
Epoch 11 loss 0.9415078373784338 valid acc 16/16
Epoch 11 loss 1.509380002873475 valid acc 15/16
Epoch 11 loss 1.3343099410330912 valid acc 14/16
Epoch 11 loss 0.978520934493768 valid acc 15/16
Epoch 11 loss 0.539826652563069 valid acc 15/16
Epoch 11 loss 0.6204780158181489 valid acc 15/16
Epoch 11 loss 1.4032743067389624 valid acc 14/16
Epoch 11 loss 1.3034538806697658 valid acc 15/16
Epoch 11 loss 1.0712598059690448 valid acc 13/16
Epoch 11 loss 2.5866012273088916 valid acc 14/16
Epoch 11 loss 1.4863204901068414 valid acc 16/16
Epoch 11 loss 2.1885626699602656 valid acc 16/16
Epoch 11 loss 1.6377207361156523 valid acc 16/16
Epoch 11 loss 1.3442318071121924 valid acc 16/16
Epoch 11 loss 0.5118337211929154 valid acc 16/16
Epoch 11 loss 1.5900502897946736 valid acc 16/16
Epoch 11 loss 0.7774897925507525 valid acc 16/16
Epoch 11 loss 0.5121077924446775 valid acc 16/16
Epoch 11 loss 0.9120369939668064 valid acc 16/16
Epoch 11 loss 0.4749884638283588 valid acc 15/16
Epoch 11 loss 1.3930640130010974 valid acc 15/16
Epoch 11 loss 0.9195641227190055 valid acc 16/16
Epoch 11 loss 0.4738145845863923 valid acc 15/16
Epoch 11 loss 0.9167147542581253 valid acc 15/16
Epoch 11 loss 0.533177225768122 valid acc 15/16
Epoch 11 loss 1.1398151796786176 valid acc 15/16
Epoch 11 loss 1.4520429386543856 valid acc 15/16
Epoch 11 loss 0.4935493049926104 valid acc 15/16
Epoch 11 loss 0.996862980739447 valid acc 15/16
Epoch 11 loss 1.8964051238668524 valid acc 16/16
Epoch 11 loss 0.7782138397017171 valid acc 16/16
Epoch 11 loss 0.7851306987141515 valid acc 15/16
Epoch 11 loss 1.1607158408628633 valid acc 16/16
Epoch 11 loss 1.2605605688188959 valid acc 15/16
Epoch 11 loss 1.094144353992649 valid acc 14/16
Epoch 11 loss 1.1329243623276635 valid acc 15/16
Epoch 11 loss 1.0693142109184026 valid acc 16/16
Epoch 11 loss 1.040292248547325 valid acc 16/16
Epoch 11 loss 0.8120321375399779 valid acc 16/16
Epoch 11 loss 0.7213399783523751 valid acc 16/16
Epoch 11 loss 0.10927136962860598 valid acc 16/16
Epoch 11 loss 0.7323782718700447 valid acc 15/16
Epoch 11 loss 1.1077162566251304 valid acc 16/16
Epoch 11 loss 0.907834742397658 valid acc 16/16
Epoch 11 loss 0.5128511439812655 valid acc 16/16
Epoch 11 loss 0.746201187247531 valid acc 16/16
Epoch 11 loss 0.973930460339941 valid acc 16/16
Epoch 11 loss 0.9292657808741105 valid acc 16/16
Epoch 11 loss 0.926225864547537 valid acc 16/16
Epoch 11 loss 0.6863370358763735 valid acc 15/16
Epoch 11 loss 0.9500708737039758 valid acc 16/16
Epoch 11 loss 0.5575490113609933 valid acc 15/16
Epoch 11 loss 0.7540382280511672 valid acc 16/16
Epoch 11 loss 1.5746156840461898 valid acc 14/16
Epoch 11 loss 1.2136865386698108 valid acc 15/16
Epoch 11 loss 1.0321919624347304 valid acc 16/16
Epoch 11 loss 0.7077829200235247 valid acc 15/16
Epoch 11 loss 1.8716982175700603 valid acc 16/16
Epoch 12 loss 0.015675676533281127 valid acc 16/16
Epoch 12 loss 1.3772898635735373 valid acc 16/16
Epoch 12 loss 1.084354611870665 valid acc 15/16
Epoch 12 loss 1.0159174789461454 valid acc 14/16
Epoch 12 loss 0.5131191673625661 valid acc 14/16
Epoch 12 loss 0.5518030431781583 valid acc 13/16
Epoch 12 loss 1.0723821881836382 valid acc 14/16
Epoch 12 loss 0.7233524188520392 valid acc 14/16
Epoch 12 loss 1.2640612584793391 valid acc 16/16
Epoch 12 loss 0.5758705321476775 valid acc 16/16
Epoch 12 loss 0.9383845406448352 valid acc 16/16
Epoch 12 loss 1.4833609059025201 valid acc 16/16
Epoch 12 loss 0.7361733688709883 valid acc 15/16
Epoch 12 loss 1.545077832614215 valid acc 14/16
Epoch 12 loss 2.1836729630045553 valid acc 16/16
Epoch 12 loss 1.2104079323741528 valid acc 15/16
Epoch 12 loss 2.4718294338098348 valid acc 15/16
Epoch 12 loss 0.9454935426130382 valid acc 16/16
Epoch 12 loss 0.7160455300826897 valid acc 16/16
Epoch 12 loss 0.6387202373450098 valid acc 16/16
Epoch 12 loss 1.178071957640525 valid acc 16/16
Epoch 12 loss 0.5651850030000045 valid acc 16/16
Epoch 12 loss 0.6568303689315856 valid acc 16/16
Epoch 12 loss 0.5565432935839745 valid acc 16/16
Epoch 12 loss 0.8403080649886373 valid acc 15/16
Epoch 12 loss 1.562920790863453 valid acc 16/16
Epoch 12 loss 0.6167185502520097 valid acc 16/16
Epoch 12 loss 0.9991375657016659 valid acc 16/16
Epoch 12 loss 0.692989334154935 valid acc 16/16
Epoch 12 loss 0.8871411436126815 valid acc 15/16
Epoch 12 loss 0.6071559968412382 valid acc 15/16
Epoch 12 loss 0.9743749315614076 valid acc 16/16
Epoch 12 loss 0.5210790650773612 valid acc 16/16
Epoch 12 loss 1.1675732932693137 valid acc 15/16
Epoch 12 loss 1.8839528729512436 valid acc 16/16
Epoch 12 loss 0.7947134139897676 valid acc 16/16
Epoch 12 loss 0.9624918073860231 valid acc 15/16
Epoch 12 loss 0.7419369181993147 valid acc 16/16
Epoch 12 loss 0.7564477154610441 valid acc 16/16
Epoch 12 loss 0.8640199944786787 valid acc 15/16
Epoch 12 loss 0.616628404627608 valid acc 15/16
Epoch 12 loss 1.0083998244136796 valid acc 15/16
Epoch 12 loss 0.5134334813072391 valid acc 16/16
Epoch 12 loss 0.5196338276777057 valid acc 15/16
Epoch 12 loss 0.8553421232334111 valid acc 15/16
Epoch 12 loss 0.251902116280676 valid acc 15/16
Epoch 12 loss 0.31718079557468726 valid acc 15/16
Epoch 12 loss 1.3865742330945285 valid acc 16/16
Epoch 12 loss 0.756186260263813 valid acc 16/16
Epoch 12 loss 0.9657491443077859 valid acc 15/16
Epoch 12 loss 0.48841276658971283 valid acc 16/16
Epoch 12 loss 1.2927941900276778 valid acc 16/16
Epoch 12 loss 1.4298168938461107 valid acc 16/16
Epoch 12 loss 0.31001924528884384 valid acc 15/16
Epoch 12 loss 0.9051603098364089 valid acc 15/16
Epoch 12 loss 0.8626191855551744 valid acc 16/16
Epoch 12 loss 0.9486349957154302 valid acc 15/16
Epoch 12 loss 0.885856305286006 valid acc 16/16
Epoch 12 loss 0.9944895635478144 valid acc 16/16
Epoch 12 loss 1.1222495911832049 valid acc 15/16
Epoch 12 loss 0.88708085350598 valid acc 16/16
Epoch 12 loss 0.45212907515995604 valid acc 16/16
Epoch 12 loss 1.4028032338276133 valid acc 15/16
Epoch 13 loss 0.13045505538284033 valid acc 16/16
Epoch 13 loss 1.0276541981172553 valid acc 15/16
Epoch 13 loss 0.9794930205052745 valid acc 16/16
Epoch 13 loss 0.5450179449200687 valid acc 15/16
Epoch 13 loss 0.35398331700979696 valid acc 14/16
Epoch 13 loss 0.46112325152445066 valid acc 14/16
Epoch 13 loss 0.9808918628253609 valid acc 14/16
Epoch 13 loss 1.1534687961687287 valid acc 14/16
Epoch 13 loss 0.5666491827173119 valid acc 15/16
Epoch 13 loss 0.37892492300629377 valid acc 15/16
Epoch 13 loss 0.9699005978815676 valid acc 15/16
Epoch 13 loss 1.1117952068945864 valid acc 13/16
Epoch 13 loss 1.3606593995065328 valid acc 15/16
Epoch 13 loss 1.2117995500548138 valid acc 12/16
Epoch 13 loss 1.8294765972048688 valid acc 13/16
Epoch 13 loss 1.5173124857183637 valid acc 14/16
Epoch 13 loss 1.8082323621715748 valid acc 16/16
Epoch 13 loss 0.8108578998247724 valid acc 16/16
Epoch 13 loss 0.8050908589319272 valid acc 16/16
Epoch 13 loss 0.44014701206005796 valid acc 16/16
Epoch 13 loss 1.1080597831239714 valid acc 16/16
Epoch 13 loss 0.3911587402615187 valid acc 16/16
Epoch 13 loss 0.21721401166540477 valid acc 16/16
Epoch 13 loss 0.270186886033349 valid acc 16/16
Epoch 13 loss 0.7455247000523613 valid acc 15/16
Epoch 13 loss 0.7529313748681612 valid acc 16/16
Epoch 13 loss 0.8732457360090427 valid acc 16/16
Epoch 13 loss 0.8322845981823013 valid acc 15/16
Epoch 13 loss 0.7016364482564748 valid acc 15/16
Epoch 13 loss 0.537969422482953 valid acc 16/16
Epoch 13 loss 1.4399020957733137 valid acc 15/16
Epoch 13 loss 1.4603606597792278 valid acc 14/16
Epoch 13 loss 0.5378525094502742 valid acc 16/16
Epoch 13 loss 0.9399105493293554 valid acc 16/16
Epoch 13 loss 1.558308878254522 valid acc 15/16
Epoch 13 loss 1.2388269977668012 valid acc 15/16
Epoch 13 loss 1.1015896808148817 valid acc 13/16
Epoch 13 loss 1.0293536473850164 valid acc 16/16
Epoch 13 loss 0.8156900989310886 valid acc 15/16
Epoch 13 loss 0.8620597398653458 valid acc 15/16
Epoch 13 loss 0.6886377164268173 valid acc 15/16
Epoch 13 loss 0.7650413634166295 valid acc 15/16
Epoch 13 loss 0.9043721846981881 valid acc 15/16
Epoch 13 loss 0.4839386732707202 valid acc 15/16
Epoch 13 loss 0.7158538518422012 valid acc 15/16
Epoch 13 loss 0.17619519797676847 valid acc 15/16
Epoch 13 loss 0.6219262253527738 valid acc 15/16
Epoch 13 loss 0.9965626462854208 valid acc 15/16
Epoch 13 loss 0.6170905567843409 valid acc 15/16
Epoch 13 loss 0.5681756278373836 valid acc 14/16
Epoch 13 loss 0.8959911261534721 valid acc 15/16
Epoch 13 loss 1.223223927539633 valid acc 16/16
Epoch 13 loss 1.424152098352643 valid acc 15/16
Epoch 13 loss 0.48629756111672834 valid acc 15/16
Epoch 13 loss 1.0641044082265376 valid acc 15/16
Epoch 13 loss 0.750012885966984 valid acc 15/16
Epoch 13 loss 1.0778625093805996 valid acc 16/16
Epoch 13 loss 0.6469124562233822 valid acc 15/16
Epoch 13 loss 1.155957282186785 valid acc 15/16
Epoch 13 loss 1.0190913324052835 valid acc 15/16
Epoch 13 loss 0.5900933474691505 valid acc 16/16
Epoch 13 loss 0.4302822971996666 valid acc 15/16
Epoch 13 loss 1.6911776473837463 valid acc 15/16
Epoch 14 loss 0.017950086342523118 valid acc 15/16
Epoch 14 loss 1.0495736192504577 valid acc 16/16
Epoch 14 loss 1.2401883141203467 valid acc 15/16
Epoch 14 loss 0.5259753980325086 valid acc 15/16
Epoch 14 loss 0.2600788259965845 valid acc 15/16
Epoch 14 loss 0.3288233638203668 valid acc 14/16
Epoch 14 loss 0.9469701960947849 valid acc 15/16
Epoch 14 loss 0.9357726094242009 valid acc 14/16
Epoch 14 loss 0.5883997525689495 valid acc 15/16
Epoch 14 loss 0.6283417642180492 valid acc 15/16
Epoch 14 loss 0.613839898639033 valid acc 16/16
Epoch 14 loss 0.9272640838053045 valid acc 14/16
Epoch 14 loss 0.9148726735716867 valid acc 14/16
Epoch 14 loss 1.5787496474758038 valid acc 13/16
Epoch 14 loss 1.7880817976456709 valid acc 14/16
Epoch 14 loss 1.1117156190291415 valid acc 15/16
Epoch 14 loss 2.0510013316574054 valid acc 15/16
Epoch 14 loss 1.3071363403989062 valid acc 15/16
Epoch 14 loss 0.8932700133254547 valid acc 14/16
Epoch 14 loss 0.6530143113380094 valid acc 14/16
Epoch 14 loss 0.7572419069743231 valid acc 15/16
Epoch 14 loss 0.45870551079233246 valid acc 14/16
Epoch 14 loss 0.3529890524401891 valid acc 15/16
Epoch 14 loss 0.8220979908323678 valid acc 14/16
Epoch 14 loss 0.6443947816205544 valid acc 14/16
Epoch 14 loss 0.5684657101092575 valid acc 14/16
Epoch 14 loss 0.7010476295724197 valid acc 15/16
Epoch 14 loss 0.44646959927798946 valid acc 15/16
Epoch 14 loss 0.6869369020587058 valid acc 15/16
Epoch 14 loss 0.6675472392750617 valid acc 15/16
Epoch 14 loss 0.7143978322694556 valid acc 15/16
Epoch 14 loss 1.263877580655882 valid acc 16/16
Epoch 14 loss 0.5188362509955667 valid acc 15/16
Epoch 14 loss 0.9125091147557557 valid acc 16/16
Epoch 14 loss 1.3209021948698914 valid acc 16/16
Epoch 14 loss 0.843260842283675 valid acc 16/16
Epoch 14 loss 1.0824870583762232 valid acc 14/16
Epoch 14 loss 0.8459496178647377 valid acc 16/16
Epoch 14 loss 0.9660702480416284 valid acc 14/16
Epoch 14 loss 0.6647897646023587 valid acc 15/16
Epoch 14 loss 0.3169744700497825 valid acc 15/16
Epoch 14 loss 0.6585646761718913 valid acc 15/16
Epoch 14 loss 0.6252072741070125 valid acc 15/16
Epoch 14 loss 0.5441929632267595 valid acc 14/16
Epoch 14 loss 1.1857261309983962 valid acc 14/16
Epoch 14 loss 0.5706738229314559 valid acc 15/16
Epoch 14 loss 0.5532554380237996 valid acc 16/16
Epoch 14 loss 0.9904076117910605 valid acc 16/16
Epoch 14 loss 0.6960506174593044 valid acc 16/16
Epoch 14 loss 0.7664711150981935 valid acc 15/16
Epoch 14 loss 0.6875903685544307 valid acc 16/16
Epoch 14 loss 0.8013867831864023 valid acc 16/16
Epoch 14 loss 0.8375321845744123 valid acc 15/16
Epoch 14 loss 0.9017206525910034 valid acc 15/16
Epoch 14 loss 0.8598250100271962 valid acc 14/16
Epoch 14 loss 0.7630905768181325 valid acc 16/16
Epoch 14 loss 0.8720354391812211 valid acc 15/16
Epoch 14 loss 0.8352472267822798 valid acc 15/16
Epoch 14 loss 0.8444241928839802 valid acc 16/16
Epoch 14 loss 1.178247231445605 valid acc 15/16
Epoch 14 loss 0.81651925392099 valid acc 16/16
Epoch 14 loss 0.7187960524556547 valid acc 16/16
Epoch 14 loss 1.0105624306050394 valid acc 16/16
Epoch 15 loss 0.011312684520787205 valid acc 16/16
Epoch 15 loss 0.6979617630305461 valid acc 16/16
Epoch 15 loss 1.0964740079500677 valid acc 15/16
Epoch 15 loss 0.5788968704085812 valid acc 15/16
Epoch 15 loss 0.18963410028895755 valid acc 15/16
Epoch 15 loss 0.5076735977014715 valid acc 15/16
Epoch 15 loss 1.0539033050473057 valid acc 15/16
Epoch 15 loss 1.0368401452479141 valid acc 15/16
Epoch 15 loss 0.7209504668893909 valid acc 15/16
Epoch 15 loss 0.26930311005219043 valid acc 15/16
Epoch 15 loss 0.5923154627243649 valid acc 15/16
Epoch 15 loss 0.6322212797940496 valid acc 15/16
Epoch 15 loss 0.782742466243479 valid acc 15/16
Epoch 15 loss 0.9157660969158379 valid acc 14/16
Epoch 15 loss 1.345498792720421 valid acc 14/16
Epoch 15 loss 1.301023833136543 valid acc 16/16
Epoch 15 loss 1.763658414620318 valid acc 16/16
Epoch 15 loss 0.7840738417438675 valid acc 16/16
Epoch 15 loss 0.49741860448958747 valid acc 16/16
Epoch 15 loss 0.541004388481707 valid acc 15/16
Epoch 15 loss 1.1239051367379282 valid acc 14/16
Epoch 15 loss 0.571003433199322 valid acc 15/16
Epoch 15 loss 0.17799194256446255 valid acc 14/16
Epoch 15 loss 0.5703306332522361 valid acc 14/16
Epoch 15 loss 0.5775997883253707 valid acc 14/16
Epoch 15 loss 0.9264731505973052 valid acc 15/16
Epoch 15 loss 0.8331291117162891 valid acc 15/16
Epoch 15 loss 0.5973021518387432 valid acc 16/16
Epoch 15 loss 0.36989199838523446 valid acc 16/16
Epoch 15 loss 0.22764333361744482 valid acc 14/16
Epoch 15 loss 0.35620397151925876 valid acc 15/16
Epoch 15 loss 1.0232395773998804 valid acc 15/16
Epoch 15 loss 0.3269704522000783 valid acc 16/16
Epoch 15 loss 0.9094608419262267 valid acc 14/16
Epoch 15 loss 1.179498447477271 valid acc 16/16
Epoch 15 loss 0.6410837402631115 valid acc 14/16
Epoch 15 loss 0.6050840518514313 valid acc 14/16
Epoch 15 loss 1.1222567319627905 valid acc 14/16
Epoch 15 loss 0.8716753643097518 valid acc 15/16
Epoch 15 loss 0.6574330076740827 valid acc 15/16
Epoch 15 loss 0.4704683418911786 valid acc 15/16
Epoch 15 loss 0.7568527572193878 valid acc 16/16
Epoch 15 loss 0.4997201255257463 valid acc 14/16
Epoch 15 loss 0.4449008694138256 valid acc 14/16
Epoch 15 loss 0.8964834479792014 valid acc 14/16
Epoch 15 loss 0.29701661338740937 valid acc 14/16
Epoch 15 loss 0.5179199021449237 valid acc 15/16
Epoch 15 loss 1.3553851654191416 valid acc 15/16
Epoch 15 loss 0.7588922349892975 valid acc 13/16
Epoch 15 loss 0.7849555445683021 valid acc 14/16
Epoch 15 loss 0.326282374366096 valid acc 15/16
Epoch 15 loss 0.6477969229223948 valid acc 15/16
Epoch 15 loss 1.3297497940365144 valid acc 15/16
Epoch 15 loss 0.6142333448478832 valid acc 15/16
Epoch 15 loss 1.0222926091098048 valid acc 15/16
Epoch 15 loss 1.1154188274913956 valid acc 15/16
Epoch 15 loss 0.4536449199154782 valid acc 15/16
Epoch 15 loss 0.682193901943027 valid acc 15/16
Epoch 15 loss 1.0485692853984871 valid acc 15/16
Epoch 15 loss 0.7432032090165785 valid acc 15/16
Epoch 15 loss 0.7760917026993549 valid acc 16/16
Epoch 15 loss 0.29099738397857056 valid acc 16/16
Epoch 15 loss 1.1536947463008969 valid acc 15/16
Epoch 16 loss 0.20448334279222305 valid acc 16/16
Epoch 16 loss 0.9908241543268917 valid acc 16/16
Epoch 16 loss 0.745249866089121 valid acc 15/16
Epoch 16 loss 0.5521494007576069 valid acc 16/16
Epoch 16 loss 0.358176474729632 valid acc 15/16
Epoch 16 loss 0.4164500946097275 valid acc 15/16
Epoch 16 loss 0.9863652195885436 valid acc 15/16
Epoch 16 loss 0.8806873784923723 valid acc 15/16
Epoch 16 loss 0.556190114412825 valid acc 15/16
Epoch 16 loss 0.7419090277522851 valid acc 16/16
Epoch 16 loss 0.6014239893587883 valid acc 16/16
Epoch 16 loss 1.186176804325602 valid acc 14/16
Epoch 16 loss 1.0982795263309952 valid acc 16/16
Epoch 16 loss 1.2927656926154878 valid acc 14/16
Epoch 16 loss 1.797823425255758 valid acc 16/16
Epoch 16 loss 1.0481052848606145 valid acc 16/16
Epoch 16 loss 1.385396746019596 valid acc 16/16
Epoch 16 loss 1.242398154649271 valid acc 16/16
Epoch 16 loss 0.8668602764156962 valid acc 15/16
Epoch 16 loss 0.22943598991328168 valid acc 15/16
Epoch 16 loss 1.420294324892954 valid acc 15/16
Epoch 16 loss 0.2933474091225385 valid acc 15/16
Epoch 16 loss 0.18867058413435053 valid acc 15/16
Epoch 16 loss 1.2148754387272267 valid acc 15/16
Epoch 16 loss 0.7073966284072779 valid acc 14/16
Epoch 16 loss 0.5510471072089973 valid acc 14/16
Epoch 16 loss 0.6590660352017113 valid acc 14/16
Epoch 16 loss 0.6595756682107011 valid acc 15/16
Epoch 16 loss 0.5024327934508312 valid acc 15/16
Epoch 16 loss 0.39453369470270805 valid acc 15/16
Epoch 16 loss 0.5344169228480374 valid acc 16/16
Epoch 16 loss 0.46362754525597444 valid acc 16/16
Epoch 16 loss 0.34649785673312405 valid acc 16/16
Epoch 16 loss 0.47130319081161964 valid acc 15/16
Epoch 16 loss 1.0249312647304833 valid acc 16/16
Epoch 16 loss 0.9128255416672111 valid acc 16/16
Epoch 16 loss 0.686760412237893 valid acc 14/16
Epoch 16 loss 0.7385299817141983 valid acc 15/16
Epoch 16 loss 0.6862281762661456 valid acc 16/16
Epoch 16 loss 0.48646560901106384 valid acc 16/16
Epoch 16 loss 0.5195935139739606 valid acc 15/16
Epoch 16 loss 0.43428719439778396 valid acc 16/16
Epoch 16 loss 0.40679639089469877 valid acc 16/16
Epoch 16 loss 0.4796493392868846 valid acc 15/16
Epoch 16 loss 1.1357136999319097 valid acc 15/16
Epoch 16 loss 0.4229241255590329 valid acc 15/16
Epoch 16 loss 0.2698532281194169 valid acc 15/16
Epoch 16 loss 1.168595675597218 valid acc 15/16
Epoch 16 loss 0.5729090172239363 valid acc 15/16
Epoch 16 loss 0.21389499640908466 valid acc 16/16
Epoch 16 loss 0.7179446549660259 valid acc 14/16
Epoch 16 loss 0.4129792758811231 valid acc 15/16
Epoch 16 loss 1.547030205129281 valid acc 14/16
Epoch 16 loss 0.5193168554869115 valid acc 15/16
Epoch 16 loss 0.8780499619565947 valid acc 16/16
Epoch 16 loss 0.8288638891753873 valid acc 16/16
Epoch 16 loss 1.1643198584462604 valid acc 16/16
Epoch 16 loss 0.9326055171035517 valid acc 16/16
Epoch 16 loss 0.9688218864746599 valid acc 15/16
Epoch 16 loss 0.8528300845100364 valid acc 16/16
Epoch 16 loss 0.7865595627856939 valid acc 16/16
Epoch 16 loss 0.8957132761462191 valid acc 16/16
Epoch 16 loss 1.2708958425405434 valid acc 15/16
Epoch 17 loss 0.3006601874751976 valid acc 16/16
Epoch 17 loss 1.42990947643116 valid acc 16/16
Epoch 17 loss 1.1540364627927076 valid acc 14/16
Epoch 17 loss 0.9669185503349956 valid acc 16/16
Epoch 17 loss 0.2472101978736226 valid acc 16/16
Epoch 17 loss 0.46499137454127165 valid acc 16/16
Epoch 17 loss 1.1350749054319875 valid acc 15/16
Epoch 17 loss 0.784680455256611 valid acc 15/16
Epoch 17 loss 0.7608988346063554 valid acc 16/16
Epoch 17 loss 0.6387160271581616 valid acc 16/16
Epoch 17 loss 0.5451265642400348 valid acc 16/16
Epoch 17 loss 1.0433631055270693 valid acc 13/16
Epoch 17 loss 1.1723066384868708 valid acc 15/16
Epoch 17 loss 1.9000498497004248 valid acc 13/16
Epoch 17 loss 0.8601099992474002 valid acc 15/16
Epoch 17 loss 0.9759296168756353 valid acc 15/16
Epoch 17 loss 1.4031920920407126 valid acc 16/16
Epoch 17 loss 0.6274976297636006 valid acc 16/16
Epoch 17 loss 0.6366592094846215 valid acc 16/16
Epoch 17 loss 0.17210049162512753 valid acc 16/16
Epoch 17 loss 1.1908352596108298 valid acc 14/16
Epoch 17 loss 0.46777584290006236 valid acc 15/16
Epoch 17 loss 0.18523167319962927 valid acc 16/16
Epoch 17 loss 0.3159550560989116 valid acc 16/16
Epoch 17 loss 0.6577503952094967 valid acc 15/16
Epoch 17 loss 0.47361659174125215 valid acc 16/16
Epoch 17 loss 0.6190267219176973 valid acc 16/16
Epoch 17 loss 0.5391573363120327 valid acc 16/16
Epoch 17 loss 0.44133057245498536 valid acc 16/16
Epoch 17 loss 0.23280189915654842 valid acc 15/16
Epoch 17 loss 0.7186171971153319 valid acc 15/16
Epoch 17 loss 0.5211179256181617 valid acc 16/16
Epoch 17 loss 0.33252749137343124 valid acc 16/16
Epoch 17 loss 0.8370385936756972 valid acc 16/16
Epoch 17 loss 1.7546260085764276 valid acc 15/16
Epoch 17 loss 0.6443908551393887 valid acc 15/16
Epoch 17 loss 0.6400108480854014 valid acc 16/16
Epoch 17 loss 1.0248195939457312 valid acc 16/16
Epoch 17 loss 0.6556331669576576 valid acc 15/16
Epoch 17 loss 0.9151026039494534 valid acc 15/16
Epoch 17 loss 0.4459943898264683 valid acc 15/16
Epoch 17 loss 0.6626359829335208 valid acc 15/16
Epoch 17 loss 0.4839497343492697 valid acc 15/16
Epoch 17 loss 0.40733953487105334 valid acc 15/16
Epoch 17 loss 0.7100908127364274 valid acc 16/16
Epoch 17 loss 0.17435957663847998 valid acc 15/16
Epoch 17 loss 0.5120628929492013 valid acc 16/16
Epoch 17 loss 0.8761288283201534 valid acc 14/16
Epoch 17 loss 0.416876227768769 valid acc 15/16
Epoch 17 loss 0.6715867762590106 valid acc 14/16
Epoch 17 loss 0.3578815558033421 valid acc 15/16
Epoch 17 loss 0.7899688070780649 valid acc 15/16
Epoch 17 loss 1.2731095322759336 valid acc 15/16
Epoch 17 loss 0.4110462299528003 valid acc 14/16
Epoch 17 loss 0.4228446791021346 valid acc 14/16
Epoch 17 loss 0.32204569560031304 valid acc 14/16
Epoch 17 loss 0.6823534876568458 valid acc 15/16
Epoch 17 loss 0.6430541836099077 valid acc 14/16
Epoch 17 loss 0.9973122826488294 valid acc 16/16
Epoch 17 loss 0.9416008869381091 valid acc 16/16
Epoch 17 loss 0.4080161427875737 valid acc 16/16
Epoch 17 loss 0.6692952766241311 valid acc 16/16
Epoch 17 loss 1.0520012760724349 valid acc 15/16
Epoch 18 loss 0.03343552623950272 valid acc 16/16
Epoch 18 loss 0.8296605278736029 valid acc 15/16
Epoch 18 loss 0.6617881806157454 valid acc 15/16
Epoch 18 loss 0.5441544309764778 valid acc 15/16
Epoch 18 loss 0.19689970124533507 valid acc 16/16
Epoch 18 loss 0.5148649459193222 valid acc 14/16
Epoch 18 loss 1.3364401912894124 valid acc 15/16
Epoch 18 loss 0.9480145652591416 valid acc 15/16
Epoch 18 loss 0.4438007529451048 valid acc 16/16
Epoch 18 loss 0.8932576402280265 valid acc 15/16
Epoch 18 loss 0.633256722358894 valid acc 16/16
Epoch 18 loss 1.2722827218319697 valid acc 15/16
Epoch 18 loss 0.4838094008088143 valid acc 15/16
Epoch 18 loss 0.422187996335894 valid acc 14/16
Epoch 18 loss 1.2227784791621887 valid acc 15/16
Epoch 18 loss 1.0291444455861882 valid acc 16/16
Epoch 18 loss 1.746462239388595 valid acc 16/16
Epoch 18 loss 0.8169439818157854 valid acc 15/16
Epoch 18 loss 0.5046533384989427 valid acc 16/16
Epoch 18 loss 0.6415795453707416 valid acc 16/16
Epoch 18 loss 0.7880532030630957 valid acc 14/16
Epoch 18 loss 0.046009865264978994 valid acc 14/16
Epoch 18 loss 0.6431178169706395 valid acc 14/16
Epoch 18 loss 0.9344814131358332 valid acc 14/16
Epoch 18 loss 0.5855041071806608 valid acc 14/16
Epoch 18 loss 0.4708088537787047 valid acc 15/16
Epoch 18 loss 0.22539247264407275 valid acc 15/16
Epoch 18 loss 0.6335188161866432 valid acc 15/16
Epoch 18 loss 0.7196705811040971 valid acc 15/16
Epoch 18 loss 0.7007621806110708 valid acc 14/16
Epoch 18 loss 0.6858280884960548 valid acc 16/16
Epoch 18 loss 0.868850391416113 valid acc 15/16
Epoch 18 loss 0.29227653619139193 valid acc 15/16
Epoch 18 loss 0.864308261057518 valid acc 16/16
Epoch 18 loss 1.2949188605711381 valid acc 15/16
Epoch 18 loss 0.7399163270741417 valid acc 16/16
Epoch 18 loss 1.019349150515418 valid acc 14/16
Epoch 18 loss 0.8621163458742737 valid acc 16/16
Epoch 18 loss 0.902599545918486 valid acc 15/16
Epoch 18 loss 1.199948841218389 valid acc 15/16
Epoch 18 loss 0.8336101192689824 valid acc 14/16
Epoch 18 loss 0.7433368821381174 valid acc 15/16
Epoch 18 loss 0.7165698262991427 valid acc 14/16
Epoch 18 loss 0.3051572137618299 valid acc 14/16
Epoch 18 loss 0.6310288463208484 valid acc 15/16
Epoch 18 loss 0.3298528621201316 valid acc 15/16
Epoch 18 loss 0.7941181348676297 valid acc 14/16
Epoch 18 loss 1.1543716855702046 valid acc 14/16
Epoch 18 loss 0.49743530343978526 valid acc 14/16
Epoch 18 loss 0.5923450978802328 valid acc 14/16
Epoch 18 loss 1.005566220339373 valid acc 15/16
Epoch 18 loss 1.2044341064930757 valid acc 15/16
Epoch 18 loss 1.0191336296582514 valid acc 14/16
Epoch 18 loss 0.5962654360669062 valid acc 15/16
Epoch 18 loss 1.0104740231933627 valid acc 14/16
Epoch 18 loss 1.1373886903253205 valid acc 15/16
Epoch 18 loss 1.350498319086368 valid acc 14/16
Epoch 18 loss 0.7074709021364518 valid acc 15/16
Epoch 18 loss 0.9973808090258567 valid acc 15/16
Epoch 18 loss 1.6163480215826975 valid acc 15/16
Epoch 18 loss 0.38118520119183374 valid acc 15/16
Epoch 18 loss 0.49854867031930417 valid acc 16/16
Epoch 18 loss 0.8768941215560366 valid acc 16/16
Epoch 19 loss 0.1624721547958367 valid acc 15/16
Epoch 19 loss 1.0907514450515166 valid acc 15/16
Epoch 19 loss 0.8363628519922157 valid acc 14/16
Epoch 19 loss 0.48914816212738654 valid acc 15/16
Epoch 19 loss 0.37634134038041633 valid acc 14/16
Epoch 19 loss 0.688504660296121 valid acc 15/16
Epoch 19 loss 0.8248330222718154 valid acc 15/16
Epoch 19 loss 0.8421680923956075 valid acc 15/16
Epoch 19 loss 0.5189342371455546 valid acc 15/16
Epoch 19 loss 0.5006417341640756 valid acc 15/16
Epoch 19 loss 0.8550423602475604 valid acc 15/16
Epoch 19 loss 1.2497857294676233 valid acc 14/16
Epoch 19 loss 0.7427413929907043 valid acc 14/16
Epoch 19 loss 0.3539025915034451 valid acc 14/16
Epoch 19 loss 1.169311671716828 valid acc 15/16
Epoch 19 loss 0.7900512626951213 valid acc 15/16
Epoch 19 loss 1.7319025903788776 valid acc 14/16
Epoch 19 loss 1.7531267868545688 valid acc 15/16
Epoch 19 loss 1.0137190224586596 valid acc 16/16
Epoch 19 loss 0.8290610654996347 valid acc 16/16
Epoch 19 loss 0.5780106245605192 valid acc 15/16
Epoch 19 loss 0.29080915895603765 valid acc 14/16
Epoch 19 loss 0.09139638506507214 valid acc 15/16
Epoch 19 loss 0.6713887643591656 valid acc 14/16
Epoch 19 loss 0.7099664828400631 valid acc 14/16
Epoch 19 loss 0.6333627224216809 valid acc 14/16
Epoch 19 loss 0.5843036798441621 valid acc 16/16
Epoch 19 loss 0.33929179924340114 valid acc 16/16
Epoch 19 loss 0.6111190785623274 valid acc 16/16
Epoch 19 loss 0.34249636313597 valid acc 15/16
Epoch 19 loss 0.27927538228115995 valid acc 15/16
Epoch 19 loss 0.9700997760921942 valid acc 14/16
Epoch 19 loss 0.2727933671903644 valid acc 14/16
Epoch 19 loss 0.643141780207874 valid acc 15/16
Epoch 19 loss 1.224835488429347 valid acc 15/16
Epoch 19 loss 0.6392291767380784 valid acc 15/16
Epoch 19 loss 0.4865078655769406 valid acc 14/16
Epoch 19 loss 0.9641197422466605 valid acc 15/16
Epoch 19 loss 0.7213604334381206 valid acc 15/16
Epoch 19 loss 0.5301927173054015 valid acc 14/16
Epoch 19 loss 0.34480489835459427 valid acc 14/16
Epoch 19 loss 0.9666975214013858 valid acc 15/16
Epoch 19 loss 0.4247698341789865 valid acc 15/16
Epoch 19 loss 0.4249042818505575 valid acc 15/16
Epoch 19 loss 0.6487002389785008 valid acc 15/16
Epoch 19 loss 0.4912717822661253 valid acc 15/16
Epoch 19 loss 0.4640534959466754 valid acc 15/16
Epoch 19 loss 0.81003715064377 valid acc 14/16
Epoch 19 loss 0.6647776544576999 valid acc 14/16
Epoch 19 loss 0.4888460724994414 valid acc 14/16
Epoch 19 loss 0.650371566836013 valid acc 14/16
Epoch 19 loss 0.7009407722440291 valid acc 15/16
Epoch 19 loss 0.9376115615904476 valid acc 14/16
Epoch 19 loss 0.817279713104436 valid acc 15/16
Epoch 19 loss 1.150935862634895 valid acc 14/16
Epoch 19 loss 0.905550034243412 valid acc 13/16
Epoch 19 loss 1.0234928305827047 valid acc 14/16
Epoch 19 loss 0.8104566292310641 valid acc 15/16
Epoch 19 loss 1.0793025795359152 valid acc 14/16
Epoch 19 loss 1.3163621698427828 valid acc 15/16
Epoch 19 loss 0.616913230842362 valid acc 15/16
Epoch 19 loss 0.6860156906678002 valid acc 15/16
Epoch 19 loss 1.1179326432828034 valid acc 15/16
Epoch 20 loss 0.022153739608051077 valid acc 15/16
Epoch 20 loss 0.8486692883549436 valid acc 15/16
Epoch 20 loss 0.4771875262796986 valid acc 15/16
Epoch 20 loss 0.45167351228222885 valid acc 16/16
Epoch 20 loss 0.2957815308146199 valid acc 16/16
Epoch 20 loss 0.29660466497130467 valid acc 16/16
Epoch 20 loss 0.8774693672527798 valid acc 15/16
Epoch 20 loss 0.4626794696746698 valid acc 15/16
Epoch 20 loss 0.8477130172951974 valid acc 15/16
Epoch 20 loss 0.3636438007029202 valid acc 16/16
Epoch 20 loss 0.4546003765057628 valid acc 16/16
Epoch 20 loss 0.788524401162704 valid acc 14/16
Epoch 20 loss 0.8924560123210948 valid acc 15/16
Epoch 20 loss 0.7902153465761008 valid acc 14/16
Epoch 20 loss 0.9384916843733663 valid acc 15/16
Epoch 20 loss 0.5350660478622902 valid acc 16/16
Epoch 20 loss 1.2388149229903298 valid acc 16/16
Epoch 20 loss 0.49197742159121816 valid acc 16/16
Epoch 20 loss 0.9611766355269203 valid acc 16/16
Epoch 20 loss 0.6842140582761547 valid acc 16/16
Epoch 20 loss 0.6957331651886061 valid acc 15/16
Epoch 20 loss 0.5531805764556823 valid acc 15/16
Epoch 20 loss 0.16931348439621363 valid acc 15/16
Epoch 20 loss 0.8090233666424377 valid acc 16/16
Epoch 20 loss 0.5750024204412453 valid acc 15/16
Epoch 20 loss 0.6205500471452933 valid acc 15/16
Epoch 20 loss 0.4880385289759164 valid acc 15/16
Epoch 20 loss 0.8128900129103631 valid acc 16/16
Epoch 20 loss 0.6022193451268594 valid acc 15/16
Epoch 20 loss 0.33226396907091216 valid acc 16/16
Epoch 20 loss 0.6629475935374152 valid acc 16/16
Epoch 20 loss 0.9510653272218033 valid acc 15/16
Epoch 20 loss 0.43483284212165885 valid acc 16/16
Epoch 20 loss 1.8059265088690784 valid acc 16/16
Epoch 20 loss 1.3253261042147317 valid acc 16/16
Epoch 20 loss 0.8138231416141584 valid acc 14/16
Epoch 20 loss 0.8219347859534174 valid acc 14/16
Epoch 20 loss 1.1003447619281213 valid acc 16/16
Epoch 20 loss 0.9274019000915326 valid acc 16/16
Epoch 20 loss 1.1788201831387641 valid acc 16/16
Epoch 20 loss 0.7050788724234132 valid acc 16/16
Epoch 20 loss 1.2787014769459677 valid acc 16/16
Epoch 20 loss 0.887757337609123 valid acc 15/16
Epoch 20 loss 0.41747606224946854 valid acc 16/16
Epoch 20 loss 0.923896647479954 valid acc 16/16
Epoch 20 loss 0.6280662724203797 valid acc 16/16
Epoch 20 loss 0.7060184827621445 valid acc 15/16
Epoch 20 loss 1.3856033803486851 valid acc 16/16
Epoch 20 loss 0.6695039262095857 valid acc 16/16
Epoch 20 loss 1.1329884355295254 valid acc 16/16
Epoch 20 loss 0.4913842161353795 valid acc 16/16
Epoch 20 loss 1.0491326699933583 valid acc 16/16
Epoch 20 loss 1.4333425646013789 valid acc 16/16
Epoch 20 loss 0.5590436790562893 valid acc 16/16
Epoch 20 loss 1.3841083603330748 valid acc 14/16
Epoch 20 loss 1.2143192957668014 valid acc 14/16
Epoch 20 loss 1.750818518979214 valid acc 16/16
Epoch 20 loss 0.7739800261695196 valid acc 16/16
Epoch 20 loss 1.135306264475874 valid acc 15/16
Epoch 20 loss 1.152018571286482 valid acc 16/16
Epoch 20 loss 0.5449239564332942 valid acc 16/16
```
<\details>
