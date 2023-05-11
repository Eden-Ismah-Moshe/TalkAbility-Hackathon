from django.db import models
from django.utils.translation import gettext_lazy as _


class Customer(models.Model):
    customer_id = models.BigAutoField(primary_key=True, editable=False)
    first_name = models.CharField(max_length=10)
    last_name = models.CharField(max_length=10)

    class Language(models.TextChoices):
        ENGLISH = 'EN', _('English')
        SPANISH = 'SP', _('Spanish')
        FRENCH = 'FR', _('French')
        GERMAN = 'GE', _('German')
        CHINESE = 'CH', _('Chinese')
        HEBREW = 'HE', _('Hebrew')
        ARABIC = 'AR', _('Arabic')
        RUSSIAN = 'RU', _('Russian')
    language = models.CharField(max_length=15, choices=Language.choices)
    age = models.IntegerField()

    class City(models.TextChoices):
        NEW_YORK = 'NY', _('New York')
        LOS_ANGELES = 'LA', _('Los Angeles')
        CHICAGO = 'CH', _('Chicago')
        HOUSTON = 'HO', _('Houston')
        PHOENIX = 'PH', _('Phoenix')
        SAN_ANTONIO = 'SA', _('San Antonio')
        SAN_DIEGO = 'SD', _('San Diego')
        DALLAS = 'DA', _('Dallas')
        SAN_JOSE = 'SJ', _('San Jose')
        AUSTIN = 'AU', _('Austin')
        JACKSONVILLE = 'JA', _('Jacksonville')
        FORT_WORTH = 'FW', _('Fort Worth')
    city = models.CharField(max_length=15, choices=City.choices)
    phone = models.CharField(max_length=15)
    email = models.EmailField(max_length=254, unique=True)


class Representative(models.Model):
    ID = models.BigAutoField(primary_key=True, editable=False)
    first_name = models.CharField(max_length=10)
    last_name = models.CharField(max_length=10)


class Request(models.Model):

    ID = models.BigAutoField(primary_key=True, editable=False)
    customer_ID = models.ForeignKey(Customer, on_delete=models.CASCADE)
    representative_ID = models.ForeignKey(Representative, on_delete=models.CASCADE)
    origin_transcript = models.CharField(max_length=200)
    target_transcript = models.CharField(max_length=200)

    class Language(models.TextChoices):
        ENGLISH = 'EN', _('English')
        SPANISH = 'SP', _('Spanish')
        FRENCH = 'FR', _('French')
        GERMAN = 'GE', _('German')
        CHINESE = 'CH', _('Chinese')
        HEBREW = 'HE', _('Hebrew')
        ARABIC = 'AR', _('Arabic')
        RUSSIAN = 'RU', _('Russian')
        OTHER = 'OTHER'
    origin_language = models.CharField(max_length=15, choices=Language.choices, default=Language.ENGLISH)

    class Status(models.TextChoices):
        PENDING = 'PENDING'
        IN_PROGRESS = 'IN_PROGRESS'
        COMPLETED = 'COMPLETED'
        CANCELLED = 'CANCELLED'
        FAILED = 'FAILED'
    status = models.CharField(max_length=15, choices=Status.choices, default=Status.PENDING)

    class Department(models.TextChoices):
        LEGAL = 'LEGAL'
        MEDICAL = 'MEDICAL'
        FINANCIAL = 'FINANCIAL'
        TECHNICAL = 'TECHNICAL'
        EDUCATIONAL = 'EDUCATIONAL'
        BUSINESS = 'BUSINESS'
        OTHER = 'OTHER'
    department = models.CharField(max_length=15, choices=Department.choices, default=Department.OTHER)
    time_of_request = models.DateTimeField('date published')
    request_summary = models.CharField(max_length=200)


