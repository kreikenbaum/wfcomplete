#! /usr/bin/env ruby

#SVM customized kernel
	websites = ARGV[0]
	trials = ARGV[1]
	`rm ./gnorm_cus_t*`
	`rm tor_result`
	`./gen_stratify #{websites} #{trials} ./matrix`

	cost = 4**5
	`echo 'gamma 2^0, cost 4^5, website #{websites}, trials #{trials}' >> ./tor_result`
	1.upto(10) do |fold|
		`./svm-train -t 4 -c #{cost} ./gnorm_cus_training_#{fold} ./gnorm_cus_trainmodel >> ./tor_result`
		`./svm-predict ./gnorm_cus_testing_#{fold} ./gnorm_cus_trainmodel ./gnorm_cus_out >> ./tor_result`
		puts "website #{websites} trials #{trials} fold #{fold} finished"
	end
	`echo '*********' >> ./tor_result`
