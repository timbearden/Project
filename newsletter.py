from datetime import datetime

class Newsletter(object):
    def __init__(self, summarizer_list):
        self.summarizer_list = summarizer_list
        self.html = ''


    def format_article(self, summarizer):
        title = summarizer.title
        summary = summarizer.summary.decode('utf-8').encode('ascii', 'xmlcharrefreplace')

        reduction = str(round(summarizer.reduction * 100, 2))
        url = summarizer.url.encode('ascii', 'xmlcharrefreplace')

        summary = summary.replace('\n', '</p><p>')
        summary = '<p>' + summary + '</p>'

        formatted = '''<h2 style="font-family:serif;"><a href="{3}">{0}</a></h2>
                        {1}
                        <b>Reduction:</b> {2}% of original sentences kept.'''.format(title, summary, reduction, url)
        return formatted


    def construct_html(self):
        summaries = [self.format_article(summarizer) for summarizer in self.summarizer_list]

        main_text = '''<table width="640" cellpadding="0" cellspacing="0" border="0" class="wrapper" bgcolor="#FFFFFF">
                            <tr>
                              <td height="10" style="font-size:10px; line-height:10px;">&nbsp;</td>
                            </tr>
                            <tr>
                              <td align="center" valign="top">

                                <table width="600" cellpadding="0" cellspacing="0" border="0" class="container">
                                  <tr>
                                    <td width="300" class="mobile" align="left" valign="top">
                                      {}
                                    </td>
                                  </tr>
                                </table>

                              </td>
                            </tr>
                            <tr>
                              <td height="10" style="font-size:10px; line-height:10px;">&nbsp;</td>
                            </tr>
                          </table>'''.format(summaries.pop(0))

        summaries = sorted(summaries, key = lambda summary: len(summary))

        for i in xrange(0, len(summaries) - 1, 2):
            main_text += '''<table width="640" cellpadding="0" cellspacing="0" border="0" class="wrapper" bgcolor="#FFFFFF">
                                <tr>
                                  <td height="10" style="font-size:10px; line-height:10px;">&nbsp;</td>
                                </tr>
                                <tr>
                                  <td align="center" valign="top">

                                    <table width="600" cellpadding="0" cellspacing="0" border="0" class="container">
                                      <tr>
                                        <td width="300" class="mobile" align="left" valign="top">
                                          {}
                                        </td>
                                        <td width="300" class="mobile" align="left" valign="top">
                                          {}
                                        </td>
                                      </tr>
                                    </table>

                                  </td>
                                </tr>
                                <tr>
                                  <td height="10" style="font-size:10px; line-height:10px;">&nbsp;</td>
                                </tr>
                              </table>'''.format(summaries[i], summaries[i + 1])

        main_text += '''<table width="640" cellpadding="0" cellspacing="0" border="0" class="wrapper" bgcolor="#FFFFFF">
                            <tr>
                              <td height="10" style="font-size:10px; line-height:10px;">&nbsp;</td>
                            </tr>
                            <tr>
                              <td align="center" valign="top">

                                <table width="600" cellpadding="0" cellspacing="0" border="0" class="container">
                                  <tr>
                                    <td width="300" class="mobile" align="left" valign="top">
                                      {}
                                    </td>
                                  </tr>
                                </table>

                              </td>
                            </tr>
                            <tr>
                              <td height="10" style="font-size:10px; line-height:10px;">&nbsp;</td>
                            </tr>
                          </table>'''.format(summaries[-1])



        header = '''<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
            <html lang="en">
            <head>
              <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
              <meta name="viewport" content="width=device-width, initial-scale=1">
              <meta http-equiv="X-UA-Compatible" content="IE=edge">

              <title>Grids Master Template</title>

              <style type="text/css">

                /* Outlines the grids, remove when sending */
                /*table td { border: 1px solid black; }*/
                /*td { border: 1px solid black; }*/
                /* CLIENT-SPECIFIC STYLES */
                body, table, td, a { -webkit-text-size-adjust: 100%; -ms-text-size-adjust: 100%; }
                table, td { mso-table-lspace: 0pt; mso-table-rspace: 0pt; }
                img { -ms-interpolation-mode: bicubic; }
                hr {
                    display: block;
                    -webkit-margin-before: 0.5em;
                    -webkit-margin-after: 0.5em;
                    -webkit-margin-start: auto;
                    -webkit-margin-end: auto;
                    border-style: inset;
                    border-width: 1px;
                }
                /* RESET STYLES */
                img { border: 0; outline: none; text-decoration: none; }
                table { border-collapse: collapse !important; }
                body { margin: 0 !important; padding: 0 !important; width: 100% !important; }
                /* iOS BLUE LINKS */
                a[x-apple-data-detectors] {
                  color: inherit !important;
                  text-decoration: none !important;
                  font-size: inherit !important;
                  font-family: inherit !important;
                  font-weight: inherit !important;
                  line-height: inherit !important;
                }
                /* ANDROID CENTER FIX */
                div[style*="margin: 16px 0;"] { margin: 0 !important; }
                /* MEDIA QUERIES */
                @media all and (max-width:639px){
                  .wrapper{ width:320px!important; padding: 0 !important; }
                  .container{ width:300px!important;  padding: 0 !important; }
                  .mobile{ width:300px!important; display:block!important; padding: 0 !important; }
                  .img{ width:100% !important; height:auto !important; }
                  *[class="mobileOff"] { width: 0px !important; display: none !important; }
                  *[class*="mobileOn"] { display: block !important; max-height:none !important; }
                }
              </style>
            </head>'''




        days = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
        months = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}
        today = datetime.now()

        today_string = "{0}, {1} {2}, {3}".format(days[today.weekday()], months[today.month], today.day, today.year)



        email_body = '''<body style="margin:0; padding:0; background-color:#F2F2F2;">
              <center>
                <table width="100%" border="0" cellpadding="0" cellspacing="0" bgcolor="#F2F2F2">
                  <tr>
                    <td align="center" valign="top">

                      <table width="640" cellpadding="0" cellspacing="0" border="0" class="wrapper" bgcolor="#FFFFFF">
                        <tr>
                          <td height="10" style="font-size:10px; line-height:10px;">&nbsp;</td>
                        </tr>
                        <tr>
                          <td align="center" valign="top">

                            <table width="600" cellpadding="0" cellspacing="0" border="0" class="container">
                              <tr>
                                <td align="center" valign="top">
                                  <img src="cid:image1">
                                </td>
                              </tr>
                              <tr>
                                <td align="center" valign="top">
                                  {0}
                                </td>
                              </tr>
                              <hr>
                            </table>

                          </td>
                        </tr>
                        <tr>
                          <td height="10" style="font-size:10px; line-height:10px;">&nbsp;</td>
                        </tr>
                      </table>

                        {1}

                    </td>
                  </tr>
                </table>
              </center>
            </body>
            </html>'''.format(today_string, main_text)

        self.html = header + email_body
