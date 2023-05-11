from django.shortcuts import render

# Create your views here.
def baseWeb(request):
    return render(request, "Main/index.html")