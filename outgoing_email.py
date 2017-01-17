import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText

def send_email(email):
    addressees = ['tbearden89@gmail.com']

    fromaddr = 'brief.news@brief-news.info'
    password = 'hKAmnt98'
    toaddr = ', '.join(addressees)

    msg = MIMEMultipart()
    msg['from'] = fromaddr
    msg['to'] = toaddr
    msg['subject'] = 'TEST EMAIL'

    msg.attach(MIMEText(email, 'plain'))

    server = smtplib.SMTP_SSL('smtp.zoho.com', 465)
    server.ehlo()
    server.login(fromaddr, password)

    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()
