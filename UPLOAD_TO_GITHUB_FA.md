# بارگذاری در گیت‌هاب

در حساب Mojtaba-jahanian مخزن GuardK ایجاد کنید.

- Repository name: GuardK
- Description: Frozen GuardK implementation, protocols, experimental results, and reproduction scripts.
- Visibility: Public
- Add README: Off
- Add .gitignore: No .gitignore
- Add license: No license

پس از Create repository، گزینه uploading an existing file را انتخاب کنید. ZIP را روی رایانه استخراج کنید و محتویات پوشه GuardK_GitHub_Ready را بارگذاری کنید، نه خود ZIP.

برای حفظ پوشه‌ها و فایل‌های نقطه‌دار، مسیر Git مطمئن‌تر است. اگر Git روی سیستم نصب است، در PowerShell وارد پوشه استخراج‌شده شوید و این دستورات را اجرا کنید:

```powershell
git init -b main
git add .
git commit -m "Add GuardK evidence and figure reproduction package"
git remote add origin https://github.com/Mojtaba-jahanian/GuardK.git
git push -u origin main
```

آدرس بالا مقصد پیشنهادی متناسب با نام حساب تصویر است؛ ایجاد مخزن و در دسترس بودن آن هنوز تأیید نشده است. اگر نام مخزن یا نام حساب متفاوت بود، دستور remote باید با URL واقعی عوض شود. در صورت درخواست نام و ایمیل در اولین commit، مشخصات واقعی حساب خود را در تنظیمات Git ثبت کنید.

سورس واقعی core و v0.4 از آرشیو G10B بازیابی و با هش قفل G9 تطبیق داده شده است. بازاجرای کامل آزمایش‌ها انجام نشده است و محدودیت‌های بازتولید در README توضیح داده شده‌اند. کاورلتر، اظهارنامه‌ها، اسکرین‌شات‌ها، کل بسته سابمیت و داده‌های خام پزشکی برای بارگذاری عمومی در این پوشه قرار نگرفته‌اند.
