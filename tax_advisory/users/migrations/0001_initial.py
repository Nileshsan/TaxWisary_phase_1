# Generated by Django 5.1.1 on 2025-03-18 10:50

import cloudinary.models
import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Advance_Tax_and_TDS',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('TDS_deducted', models.BooleanField(default=False)),
                ('TDS_deducted_amount', models.FloatField(default=0)),
                ('password_Income_Tax_ID', models.CharField(blank=True, max_length=100, null=True)),
                ('Advance_Tax_paid', models.BooleanField(default=False)),
                ('Advance_Tax_paid_amount', models.FloatField(default=0)),
                ('additional_TDS_deducted_Fixed_Deposits_Interest_or_RentPayments', models.BooleanField(default=False)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Bank_Account_Details',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Bank_Name', models.CharField(blank=True, max_length=100, null=True)),
                ('Bank_Accounts_number', models.CharField(blank=True, max_length=100, null=True)),
                ('Bank_IFSC', models.CharField(blank=True, max_length=100, null=True)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='CapitalGainAndOtherIncome',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Conformation_on_capital_asset_sold', models.BooleanField(default=False)),
                ('Determine_capital_gain_criteria', models.CharField(blank=True, max_length=100, null=True)),
                ('was_Real_Estate_sold_before23rdJul2024', models.BooleanField(default=False)),
                ('asset_type', models.CharField(blank=True, max_length=100, null=True)),
                ('profit_real_estate', models.FloatField(default=0)),
                ('was_stock_listen_on_stock_exchange', models.BooleanField(default=False)),
                ('when_stock_purchased', models.DateField()),
                ('when_stock_sold', models.DateField()),
                ('stock_purchase_price', models.FloatField(default=0)),
                ('stock_sale_price', models.FloatField(default=0)),
                ('Eligible_preferential_tax_treatment', models.BooleanField(default=False)),
                ('profit_stocks', models.FloatField(default=0)),
                ('involved_in_Other_assets', models.BooleanField(default=False)),
                ('when_asset_purchased', models.DateField()),
                ('when_asset_sold', models.DateField()),
                ('asset_purchase_price', models.FloatField(default=0)),
                ('asset_sale_price', models.FloatField(default=0)),
                ('profit_other_assets', models.FloatField(default=0)),
                ('total_capital_gain', models.FloatField(default=0)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Deductions',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('epf_amount', models.FloatField(default=0)),
                ('life_insurance_amount', models.FloatField(default=0)),
                ('mutual_fund_value', models.FloatField(default=0)),
                ('ppf_value', models.FloatField(default=0)),
                ('nsc_value', models.FloatField(default=0)),
                ('home_loan_principal', models.FloatField(default=0)),
                ('child_tuition_fees', models.FloatField(default=0)),
                ('other_80C_amount', models.FloatField(default=0)),
                ('Deduction_A', models.FloatField(default=0)),
                ('Deduction_B1', models.FloatField(default=0)),
                ('Deduction_B2', models.FloatField(default=0)),
                ('Deduction_B', models.FloatField(default=0)),
                ('education_loan_interest', models.FloatField(default=0)),
                ('Deduction_C', models.FloatField(default=0)),
                ('nps_amount', models.FloatField(default=0)),
                ('Deduction_D', models.FloatField(default=0)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='HousePropertyIncome',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Number_of_properties', models.CharField(blank=True, max_length=100, null=True)),
                ('Are_Properties_rented', models.CharField(blank=True, null=True, verbose_name=bool)),
                ('rent_income', models.FloatField(default=0)),
                ('Municipal_Tax', models.FloatField(default=0)),
                ('Maintenance_Charges', models.FloatField(default=0)),
                ('Rental_Income', models.FloatField(default=0)),
                ('Secondary_Income', models.FloatField(default=0)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='IncomeDetails',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('employer_name', models.CharField(blank=True, max_length=100, null=True)),
                ('tan', models.CharField(blank=True, max_length=10, null=True)),
                ('salary_income', models.FloatField(default=0)),
                ('house_rent_applied', models.BooleanField(default=False)),
                ('house_rent_amount', models.FloatField(blank=True, null=True)),
                ('landlord_pan', models.CharField(blank=True, max_length=10, null=True)),
                ('house_rent_exemption', models.FloatField(default=0)),
                ('travel_allowance_applied', models.BooleanField(default=False)),
                ('travel_allowance_amount', models.FloatField(default=0)),
                ('other_income_applied', models.BooleanField(default=False)),
                ('other_income_source', models.CharField(blank=True, max_length=100, null=True)),
                ('other_income_amount', models.FloatField(default=0)),
                ('proprietary_income', models.FloatField(default=0)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='incomes_on_investment',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('home_loan_interest', models.FloatField(default=0)),
                ('Deduction_H', models.FloatField(default=0)),
                ('interest_on_savings', models.FloatField(default=0)),
                ('Deduction_F', models.FloatField(default=0)),
                ('interest_on_FD_senior_citizen', models.FloatField(default=0)),
                ('Deduction_G', models.FloatField(default=0)),
                ('Divident', models.FloatField(default=0)),
                ('primary_income', models.FloatField(default=0)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='OtherDisclosure',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('foreign_Bank_Accounts', models.BooleanField(default=False)),
                ('foreign_Bank_Accounts_details', models.CharField(blank=True, max_length=100, null=True)),
                ('foreign_Assets', models.BooleanField(default=False)),
                ('foreign_Assets_details', models.CharField(blank=True, max_length=100, null=True)),
                ('foreign_Investment', models.BooleanField(default=False)),
                ('foreign_Investment_details', models.CharField(blank=True, max_length=100, null=True)),
                ('foreign_Income', models.BooleanField(default=False)),
                ('foreign_Income_details', models.CharField(blank=True, max_length=100, null=True)),
                ('Received_Gifts_above50k', models.BooleanField(default=False)),
                ('Received_Gifts_above50k_details', models.CharField(blank=True, max_length=100, null=True)),
                ('Gift_taxable_amount', models.FloatField(default=0)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='TaxData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('income', models.FloatField()),
                ('deductions', models.FloatField()),
                ('home_loan', models.FloatField()),
                ('tax_report', models.URLField(blank=True, null=True)),
                ('status', models.CharField(default='Pending', max_length=50)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='TaxReport',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('pdf_url', cloudinary.models.CloudinaryField(blank=True, max_length=255, null=True, verbose_name='file')),
                ('status', models.CharField(default='Pending', max_length=50)),
                ('total_income', models.FloatField(default=0)),
                ('total_deductions', models.FloatField(default=0)),
                ('net_taxable_income', models.FloatField(default=0)),
                ('tax_amount', models.FloatField(default=0)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='TempAdvanceTaxAndTDS',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('TDS_deducted', models.BooleanField(default=False)),
                ('TDS_deducted_amount', models.FloatField(default=0)),
                ('password_Income_Tax_ID', models.CharField(blank=True, max_length=100, null=True)),
                ('Advance_Tax_paid', models.BooleanField(default=False)),
                ('Advance_Tax_paid_amount', models.FloatField(default=0)),
                ('additional_TDS_deducted_Fixed_Deposits_Interest_or_RentPayments', models.BooleanField(default=False)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='TempBankAccountDetails',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Bank_Name', models.CharField(blank=True, max_length=100, null=True)),
                ('Bank_Accounts_number', models.CharField(blank=True, max_length=100, null=True)),
                ('Bank_IFSC', models.CharField(blank=True, max_length=100, null=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='TempCapitalGainAndOtherIncome',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Conformation_on_capital_asset_sold', models.BooleanField(default=False)),
                ('Determine_capital_gain_criteria', models.CharField(blank=True, max_length=100, null=True)),
                ('was_Real_Estate_sold_before23rdJul2024', models.BooleanField(default=False)),
                ('asset_type', models.CharField(blank=True, max_length=100, null=True)),
                ('profit_real_estate', models.FloatField(default=0)),
                ('was_stock_listen_on_stock_exchange', models.BooleanField(default=False)),
                ('when_stock_purchased', models.DateField()),
                ('when_stock_sold', models.DateField()),
                ('stock_purchase_price', models.FloatField(default=0)),
                ('stock_sale_price', models.FloatField(default=0)),
                ('Eligible_preferential_tax_treatment', models.BooleanField(default=False)),
                ('profit_stocks', models.FloatField(default=0)),
                ('involved_in_Other_assets', models.BooleanField(default=False)),
                ('when_asset_purchased', models.DateField()),
                ('when_asset_sold', models.DateField()),
                ('asset_purchase_price', models.FloatField(default=0)),
                ('asset_sale_price', models.FloatField(default=0)),
                ('profit_other_assets', models.FloatField(default=0)),
                ('total_capital_gain', models.FloatField(default=0)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='TempDeductions',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('epf_amount', models.FloatField(default=0)),
                ('life_insurance_amount', models.FloatField(default=0)),
                ('mutual_fund_value', models.FloatField(default=0)),
                ('ppf_value', models.FloatField(default=0)),
                ('nsc_value', models.FloatField(default=0)),
                ('home_loan_principal', models.FloatField(default=0)),
                ('child_tuition_fees', models.FloatField(default=0)),
                ('other_80C_amount', models.FloatField(default=0)),
                ('Deduction_A', models.FloatField(default=0)),
                ('Deduction_B1', models.FloatField(default=0)),
                ('Deduction_B2', models.FloatField(default=0)),
                ('Deduction_B', models.FloatField(default=0)),
                ('education_loan_interest', models.FloatField(default=0)),
                ('Deduction_C', models.FloatField(default=0)),
                ('nps_amount', models.FloatField(default=0)),
                ('Deduction_D', models.FloatField(default=0)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='TempHousePropertyIncome',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Number_of_properties', models.CharField(blank=True, max_length=100, null=True)),
                ('Are_Properties_rented', models.BooleanField(default=False)),
                ('rent_income', models.FloatField(default=0)),
                ('Municipal_Tax', models.FloatField(default=0)),
                ('Maintenance_Charges', models.FloatField(default=0)),
                ('Rental_Income', models.FloatField(default=0)),
                ('Secondary_Income', models.FloatField(default=0)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='TempIncomeDetails',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('employer_name', models.CharField(blank=True, max_length=100, null=True)),
                ('tan', models.CharField(blank=True, max_length=10, null=True)),
                ('salary_income', models.FloatField(default=0)),
                ('house_rent_applied', models.BooleanField(default=False)),
                ('house_rent_amount', models.FloatField(blank=True, null=True)),
                ('landlord_pan', models.CharField(blank=True, max_length=10, null=True)),
                ('house_rent_exemption', models.FloatField(default=0)),
                ('travel_allowance_applied', models.BooleanField(default=False)),
                ('travel_allowance_amount', models.FloatField(default=0)),
                ('other_income_applied', models.BooleanField(default=False)),
                ('other_income_source', models.CharField(blank=True, max_length=100, null=True)),
                ('other_income_amount', models.FloatField(default=0)),
                ('proprietary_income', models.FloatField(default=0)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='TempIncomesOnInvestment',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('home_loan_interest', models.FloatField(default=0)),
                ('Deduction_H', models.FloatField(default=0)),
                ('interest_on_savings', models.FloatField(default=0)),
                ('Deduction_F', models.FloatField(default=0)),
                ('interest_on_FD_senior_citizen', models.FloatField(default=0)),
                ('Deduction_G', models.FloatField(default=0)),
                ('Divident', models.FloatField(default=0)),
                ('primary_income', models.FloatField(default=0)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='TempOtherDisclosure',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('foreign_Bank_Accounts', models.BooleanField(default=False)),
                ('foreign_Bank_Accounts_details', models.CharField(blank=True, max_length=100, null=True)),
                ('foreign_Assets', models.BooleanField(default=False)),
                ('foreign_Assets_details', models.CharField(blank=True, max_length=100, null=True)),
                ('foreign_Investment', models.BooleanField(default=False)),
                ('foreign_Investment_details', models.CharField(blank=True, max_length=100, null=True)),
                ('foreign_Income', models.BooleanField(default=False)),
                ('foreign_Income_details', models.CharField(blank=True, max_length=100, null=True)),
                ('Received_Gifts_above50k', models.BooleanField(default=False)),
                ('Received_Gifts_above50k_details', models.CharField(blank=True, max_length=100, null=True)),
                ('Gift_taxable_amount', models.FloatField(default=0)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='TempUserData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('full_name', models.CharField(default='Jhon Doe', max_length=255)),
                ('pan', models.CharField(default='ABCDE1234Z', max_length=10)),
                ('uid', models.CharField(default='123456789012', max_length=12)),
                ('dob', models.DateField(default='2000-01-01')),
                ('phone', models.CharField(default='1234567890', max_length=10)),
                ('email', models.EmailField(default='user@server.com', max_length=254)),
                ('address', models.TextField(default='flat no. 123, street 456, city, state, country')),
                ('employment', models.CharField(default='default', max_length=50)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='TempUserProfile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('full_name', models.CharField(max_length=100)),
                ('pan', models.CharField(max_length=10, unique=True)),
                ('uid', models.CharField(max_length=12, unique=True)),
                ('dob', models.DateField()),
                ('contact_info', models.CharField(max_length=100)),
                ('address', models.TextField()),
                ('employment_type', models.CharField(choices=[('Salaried', 'Salaried'), ('Business', 'Business')], max_length=50)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='UserProfile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('full_name', models.CharField(max_length=100)),
                ('pan', models.CharField(max_length=10, unique=True)),
                ('uid', models.CharField(max_length=12, unique=True)),
                ('dob', models.DateField()),
                ('contact_info', models.CharField(max_length=100)),
                ('address', models.TextField()),
                ('employment_type', models.CharField(choices=[('Salaried', 'Salaried'), ('Business', 'Business')], max_length=50)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
