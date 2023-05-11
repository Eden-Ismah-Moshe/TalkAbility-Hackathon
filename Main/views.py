from django.shortcuts import render

# Create your views here.
def baseWeb(request):
    # Get logged in users
    # user_profile = Profile.objects.all()[0]
    #
    # context = {
    #     # 'user_profile': request.user.profile
    #     'user_profile': user_profile
    # }
    return render(request, 'Main/index.html')
