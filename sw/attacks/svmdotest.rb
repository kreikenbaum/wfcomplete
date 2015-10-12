#! /usr/bin/env ruby

#SVM customized kernel
	gamma = 2**0
	website = 100
	`rm cl.train cl.model cl.test cl.out`
	`./clgen_stratify cltor_matrix 36 40`

	cost = 4**5
	`~/Downloads/libsvm/svm-train -t 4 -c #{cost} ./cl.train ./cl.model >> ./clLev_acc`
	`~/Downloads/libsvm/svm-predict ./cl.test ./cl.model ./cl.out >> ./clLev_acc`

	`echo '*********' >> ./clLev_acc`
