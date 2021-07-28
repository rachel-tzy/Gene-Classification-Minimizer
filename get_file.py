import paramiko

# create ssh
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# connect to the server
ssh.connect(hostname='hamlet.math.toronto.edu', port=443, username='ztao', password='banana6rock12')

# Setup sftp connection and transmit this script
sftp = ssh.open_sftp()
path_local = 'C:\\Users\\TZY\PycharmProjects\\Minimizer\\data\\'
path_remote = '/tmp/tzy/Minimizer/data/'
sftp.get(path_remote + 'Mgnify-5-genus-prediction-50to200_history_abword.npy',
         path_local + 'Mgnify-5-genus-prediction-50to200_history_abword.npy')

sftp.close()

ssh.close()
