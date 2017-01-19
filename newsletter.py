class Newsletter(object):
    def __init__(self, summarizer_list):
        self.summarizer_list = summarizer_list
        self.html = ''


    def format_article(self, summarizer):
        title = summarizer.title.decode('utf-8').encode('ascii', 'xmlcharrefreplace')
        summary = summarizer.summary.decode('utf-8').encode('ascii', 'xmlcharrefreplace')
        reduction = str(round(summarizer.reduction * 100, 2))
        url = summarizer.url.encode('ascii', 'xmlcharrefreplace')
        summary = summary.replace('\n', '</p><p>')
        summary = '<p>' + summary + '</p>'
        formatted = '''<h2>{0}</h2>
                        {1}
                        <b>Reduction:</b> {2}% of original sentences kept.
                        <b><a href="{3}">Link to full article</a></b>'''.format(title, summary, reduction, url)
        return formatted


    def construct_html(self):
        summaries = [self.format_article(summarizer) for summarizer in self.summarizer_list]

        main_text = ''
        for summary in summaries:
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
                              </table>'''.format(summary)



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


        email_body = '''
        <body style="margin:0; padding:0; background-color:#F2F2F2;">
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
                        </table>

                      </td>
                    </tr>
                    <tr>
                      <td height="10" style="font-size:10px; line-height:10px;">&nbsp;</td>
                    </tr>
                  </table>

                    {}

                </td>
              </tr>
            </table>
          </center>
        </body>
        </html>'''.format(main_text)

        self.html = header + email_body
