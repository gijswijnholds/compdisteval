from compdisteval.code.util.read import readShelve, saveShelve
from compdisteval.code.util.paths import joinPaths
from compdisteval.code.util.paths \
    import word2vecTensorsFolder, gloveTensorsFolder
from compdisteval.code.util.paths \
    import fastTextTensorsFolder, tensorSpaceFolder


def mergeShelves(fn1, fn2, outFN):
    shelf1 = readShelve(fn1)
    shelf2 = readShelve(fn2)
    newShelf = {}
    assert set(shelf1.keys()).intersection(set(shelf2.keys())) == set([])
    for k1 in shelf1:
        newShelf[k1] = shelf1[k1]
    for k2 in shelf2:
        newShelf[k2] = shelf2[k2]
    saveShelve(newShelf, outFN)


word2vecTensorFN1 = joinPaths(word2vecTensorsFolder, 'tensors_from_file_gs2011ks2014verbs.txt_myWord2VecModellemmas_shelved.shelve')
word2vecTensorFN2 = joinPaths(word2vecTensorsFolder, 'tensors_from_file_thesis_dims_300_ml2008ks2013verbs.txt_myWord2VecModellemmas_shelved.shelve')
word2vecTensorOutFN = joinPaths(word2vecTensorsFolder, 'tensors_from_file_thesis_dims_300_ml2008gs2011ks2013ks2014verbs_myWord2VecModellemmas_shelved.shelve')

gloveTensorFN1 = joinPaths(gloveTensorsFolder, 'tensors_from_file_gs2011ks2014verbs.txt_vspace_vectors_ukwacky.shelve')
gloveTensorFN2 = joinPaths(gloveTensorsFolder, 'tensors_from_file_thesis_dims_300_ml2008ks2013verbs.txt_vspace_vectors_ukwacky.shelve')
gloveTensorOutFN = joinPaths(gloveTensorsFolder, 'tensors_from_file_thesis_dims_300_ml2008gs2011ks2013ks2014verbs_vspace_vectors_ukwacky.shelve')

fasttextTensorFN1 = joinPaths(fastTextTensorsFolder, 'tensors_from_file_gs2011ks2014verbs.txt_myFastTextModellemmas_shelved.shelve')
fasttextTensorFN2 = joinPaths(fastTextTensorsFolder, 'tensors_from_file_thesis_dims_300_ml2008ks2013verbs.txt_myFastTextModellemmas_shelved.shelve')
fasttextTensorOutFN = joinPaths(fastTextTensorsFolder, 'tensors_from_file_thesis_dims_300_ml2008gs2011ks2013ks2014verbs_myFastTextModellemmas_shelved.shelve')

countTensorFN1 = joinPaths(tensorSpaceFolder, 'tensors_dims_2000_from_file_gs2011ks2014verbs.txt_vspace_gijs_raw_CW=5_DIMS=10000_mp_NORM_ppmi.shelve')
countTensorFN2 = joinPaths(tensorSpaceFolder, 'tensors_thesis_dims_2000_from_file_ml2008ks2013verbs.txt_vspace_gijs_raw_CW=5_DIMS=10000_mp_NORM_ppmi.shelve')
countTensorOutFN = joinPaths(tensorSpaceFolder, 'tensors_from_file_thesis_dims_300_ml2008gs2011ks2013ks2014verbs_vspace_gijs_raw_CW=5_DIMS=10000_mp_NORM_ppmi.shelve')
