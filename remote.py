import paramiko

# create ssh
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# connect to the server
ssh.connect(hostname='hamlet.math.toronto.edu', port=443, username='ztao', password='banana6rock12')

# Setup sftp connection and transmit this script
sftp = ssh.open_sftp()

path_local = 'C:\\Users\\TZY\\PycharmProjects\\Minimizer\\'
path_remote = '/tmp/tzy/Minimizer/'
sftp.put(path_local+'data_handling.py', path_remote+'data_handling.py')
sftp.put(path_local+'model.py', path_remote+'model.py')
sftp.put(path_local+'train_find_ab.py', path_remote+'train_find_ab.py')
sftp.put(path_local+'predict_ab_word.py', path_remote+'predict_ab_word.py')
sftp.put(path_local+'train_minimizer.py', path_remote+'train_minimizer.py')

sftp.put(path_local+'data_handling_abword.py', path_remote+'data_handling_abword.py')
sftp.put(path_local+'train_abword_svm.py', path_remote+'train_abword_svm.py')
sftp.put(path_local+'predict_abword_svm.py', path_remote+'predict_abword_svm.py')


# path_local = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\' \
#                   '10genus_73\\'
# path_remote = '/tmp/tzy/Metagenomic-Data/Mgnify/10genus_73/'
# sftp.put(path_local+'train\\10genus_label_dict.npy', path_remote+'train/10genus_label_dict.npy')
# sftp.put(path_local+'test\\10genus_label_dict.npy', path_remote+'test/10genus_label_dict.npy')
# sftp.put(path_local+'test\\10genus_dict.npy', path_remote+'test/10genus_dict.npy')
# sftp.put(path_local+'train\\10genus_dict.npy', path_remote+'train/10genus_dict.npy')
#


sftp.close()
stdout = ssh.exec_command('python3/tmp/test.py')[1]
for line in stdout:
    print(line)

ssh.close()
