# Technical Troubleshooting Guide

**Category**: Technical Support  
**Last Updated**: July 2026  
**Applies To**: All SmartTech products

---

## Common Error Codes

### Error E-100: Boot Failure

**What it means**: The laptop cannot find a bootable operating system.

**Steps to fix**:
1. Restart the laptop and press **F2** during boot to enter BIOS
2. Check that the internal SSD is listed under "Boot Devices"
3. If the SSD is listed, set it as the first boot device
4. If the SSD is NOT listed, the drive may be disconnected or failed
5. Try a hard reset: power off, remove charger, hold power for 30 seconds, reconnect and restart
6. If the error persists, contact support — your SSD may need replacement under warranty

### Error E-200: Thermal Throttling Warning

**What it means**: The laptop's CPU temperature exceeded safe limits and performance has been reduced to prevent damage.

**Steps to fix**:
1. Make sure all air vents are clear and unobstructed
2. Use the laptop on a flat, hard surface — not on a bed, sofa, or blanket
3. Close resource-heavy applications (check Task Manager for CPU usage)
4. Clean the vents with compressed air (hold the can upright, spray in short bursts)
5. If you have a GameStation 16, open SmartTech Control Center and set fan mode to "Performance"
6. In hot Indian summers (40°C+), use the laptop in an air-conditioned room for best performance
7. If the problem occurs frequently under normal use, contact support for a thermal paste replacement (covered under warranty)

### Error E-301: Wi-Fi Connection Failed

**What it means**: The laptop cannot connect to a Wi-Fi network.

**Steps to fix**:
1. Make sure Wi-Fi is enabled (check the Wi-Fi icon in the system tray)
2. Toggle airplane mode off if it's enabled
3. Restart your router/modem — unplug for 30 seconds, plug back in
4. Forget the Wi-Fi network and reconnect: Settings > Network > Wi-Fi > Manage Known Networks
5. Update the Wi-Fi driver: Device Manager > Network Adapters > right-click > Update Driver
6. Run the built-in network troubleshooter: Settings > Troubleshoot > Network
7. If using a JioFiber, Airtel Xstream, or BSNL connection, check if the issue is with your ISP first
8. If the issue persists on all networks, the Wi-Fi module may need repair — contact support

### Error E-401: Payment Processing Failed

**What it means**: Your payment could not be processed during checkout on smarttech.in.

**Steps to fix**:
1. Verify your card number, expiration date, and CVV are entered correctly
2. For UPI payments, make sure your UPI app is updated and has sufficient balance
3. For net banking, check if your bank's servers are under maintenance (common during late-night hours)
4. Check that your card has sufficient funds or credit limit available
5. If using international card, make sure international transactions are enabled
6. Try a different payment method (UPI is most reliable)
7. Disable your VPN if you're using one — banks may flag VPN transactions
8. Contact your bank to ensure they haven't blocked the transaction
9. If none of the above works, contact SmartTech support and we can help with the order

### Error E-500: Driver Installation Failed

**What it means**: A driver update downloaded from smarttech.in/drivers failed to install.

**Steps to fix**:
1. Restart your laptop and try the installation again
2. Run the installer as Administrator (right-click > Run as administrator)
3. Temporarily disable your antivirus software during installation
4. Make sure you downloaded the driver for the correct product model
5. Check that your Windows version is supported (the driver page lists compatible OS versions)
6. If the issue persists, use Windows Device Manager to manually install the driver

---

## General Troubleshooting Steps

Before contacting support, try these universal steps:

### For Hardware Issues
1. **Restart** — Fixes 50% of issues
2. **Update drivers** — Visit smarttech.in/drivers and enter your serial number
3. **Run SmartTech Diagnostics** — Pre-installed on all laptops (search "SmartTech Diagnostics" in Start Menu)
4. **Check for Windows updates** — Settings > Windows Update
5. **Hard reset** — Power off, unplug charger, hold power button for 30 seconds

### For Software Issues
1. **Restart** — Always the first step
2. **Check for updates** — Both Windows and the specific application
3. **Reinstall the application** — Uninstall, restart, then install fresh
4. **Create a new user profile** — If the issue is profile-specific
5. **System Restore** — Roll back to a previous working state

### For Network Issues
1. **Restart your router** — Unplug for 30 seconds, plug back in
2. **Forget and reconnect** — Remove the saved network and set it up again
3. **Try a different network** — Test on mobile hotspot (Jio, Airtel, Vi)
4. **Use Ethernet** — If available, connect via the RJ45 port (GameStation 16) or USB-C hub
5. **Contact your ISP** — If the issue is with all devices, it's likely your internet connection (JioFiber, Airtel, BSNL, ACT, etc.)

### For Power Issues
1. **Check your power supply** — Use a UPS or voltage stabilizer if you experience frequent power cuts
2. **Use the original charger** — Third-party chargers may not provide sufficient wattage
3. **Check the wall socket** — Try a different socket; loose connections can cause intermittent charging
4. **Surge protection** — Indian power grids can have voltage spikes during monsoon season; use a surge protector

---

## Contacting Technical Support

If troubleshooting doesn't resolve your issue:

| Channel | Availability | Best For |
|---------|-------------|----------|
| Live Chat (smarttech.in) | 24/7 | Quick questions, account issues |
| Phone (1800-123-4567, toll-free) | Mon-Sat 9AM-9PM IST | Complex hardware issues, warranty claims |
| Email (support@smarttech.in) | Response within 24 hours | Non-urgent issues, sending screenshots/logs |
| WhatsApp (+91-98765-43210) | Mon-Sat 9AM-9PM IST | Quick support with photo/video sharing |
| Community Forum (forum.smarttech.in) | 24/7 (community-moderated) | Tips, tricks, and peer support |

When contacting support, please have ready:
- Your **serial number** (bottom of laptop or Settings > About)
- Your **order number** (from confirmation email/SMS)
- A description of the issue and any **error codes**
- Steps you've already tried
