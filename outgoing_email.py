import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEImage import MIMEImage


def send_email(html):
    addressees = ['tbearden89@gmail.com']

    fromaddr = 'brief.news@brief-news.info'
    password = 'hKAmnt98'
    toaddr = ', '.join(addressees)

    msg = MIMEMultipart()
    msg['from'] = fromaddr
    msg['to'] = toaddr
    msg['subject'] = 'TEST EMAIL'

    # msg.attach(MIMEText(text, 'plain'))
    msg.attach(MIMEText(html, 'html'))

    fp = open('BriefNewsLogo.png', 'rb')
    msgImage = MIMEImage(fp.read())
    fp.close()

    msgImage.add_header('Content-ID', '<image1>')
    msg.attach(msgImage)


    server = smtplib.SMTP_SSL('smtp.zoho.com', 465)
    server.ehlo()
    server.login(fromaddr, password)

    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()
