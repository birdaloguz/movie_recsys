from django.shortcuts import redirect

def login_redirect(request):
    return redirect('/account/login')

def home_redirect(request):
    #redirect blank url to homepage
    return redirect("/home/")