# first see if possible to just use counter and 'cat' those files to background.json
# would preserve name info
ls > bg_list                                                                    
mkdir bg                                                                        
cd bg/                                                                          
for i in $(ls .. | grep @); do ln ../$i ./background@$(echo $i | cut -d '@' -f \
2); done                                                                        
w_o_errs_counter.sh
cd ..                                                                           
mv bg/background.json .                                                         
head bg_list                                                                    
rm -rf bg/*                                                                     
rm *@*                                                                          
ls                                                                              
rmdir bg/                                                                       
cd ..
