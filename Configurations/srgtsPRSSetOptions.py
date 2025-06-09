def srgtsPRSSetOptions(P=None, T=None, PRS_Degree=2, PRS_Regression='Full'):


    # data
    srgtOPT = {
        'SRGT': 'PRS',
        'P': P,
        'T': T,
    }

    # options
    if P is not None and T is not None:
        srgtOPT['PRS_Degree'] = PRS_Degree
        srgtOPT['PRS_Regression'] = PRS_Regression
    else:
        srgtOPT['PRS_Degree'] = 2
        srgtOPT['PRS_Regression'] = 'Full'

    return srgtOPT

