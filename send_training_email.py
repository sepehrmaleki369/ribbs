#!/usr/bin/env python3
"""
Send email notification when training is complete
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import os
from datetime import datetime
import glob

def send_training_complete_email(
    recipient_email: str,
    training_summary: dict,
    sender_email: str = None,
    sender_password: str = None,
    smtp_server: str = "smtp.gmail.com",
    smtp_port: int = 587
):
    """
    Send email notification when training is complete
    
    Args:
        recipient_email: Email address to send notification to
        training_summary: Dictionary with training results (epochs, losses, etc.)
        sender_email: Gmail address (if None, will use recipient_email)
        sender_password: App password for Gmail (required for Gmail SMTP)
        smtp_server: SMTP server address
        smtp_port: SMTP port
    """
    
    if sender_email is None:
        sender_email = recipient_email
    
    # Create message
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"🎉 Training Complete - {training_summary.get('model_name', 'Model')}"
    msg['From'] = sender_email
    msg['To'] = recipient_email
    
    # Create HTML content
    html_content = f"""
    <html>
      <head></head>
      <body style="font-family: Arial, sans-serif; padding: 20px; background-color: #f5f5f5;">
        <div style="max-width: 800px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
          <h1 style="color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px;">
            🎉 Training Complete!
          </h1>
          
          <div style="margin: 20px 0;">
            <h2 style="color: #34495e;">Training Summary</h2>
            <table style="width: 100%; border-collapse: collapse; margin: 15px 0;">
              <tr style="background-color: #ecf0f1;">
                <td style="padding: 12px; border: 1px solid #bdc3c7; font-weight: bold;">Model Name</td>
                <td style="padding: 12px; border: 1px solid #bdc3c7;">{training_summary.get('model_name', 'N/A')}</td>
              </tr>
              <tr>
                <td style="padding: 12px; border: 1px solid #bdc3c7; font-weight: bold;">Total Epochs</td>
                <td style="padding: 12px; border: 1px solid #bdc3c7;">{training_summary.get('total_epochs', 'N/A')}</td>
              </tr>
              <tr style="background-color: #ecf0f1;">
                <td style="padding: 12px; border: 1px solid #bdc3c7; font-weight: bold;">Start Epoch</td>
                <td style="padding: 12px; border: 1px solid #bdc3c7;">{training_summary.get('start_epoch', 'N/A')}</td>
              </tr>
              <tr>
                <td style="padding: 12px; border: 1px solid #bdc3c7; font-weight: bold;">End Epoch</td>
                <td style="padding: 12px; border: 1px solid #bdc3c7;">{training_summary.get('end_epoch', 'N/A')}</td>
              </tr>
              <tr style="background-color: #ecf0f1;">
                <td style="padding: 12px; border: 1px solid #bdc3c7; font-weight: bold;">Loss Type</td>
                <td style="padding: 12px; border: 1px solid #bdc3c7;">{training_summary.get('loss_type', 'N/A')}</td>
              </tr>
              <tr>
                <td style="padding: 12px; border: 1px solid #bdc3c7; font-weight: bold;">Final Train Loss</td>
                <td style="padding: 12px; border: 1px solid #bdc3c7; color: #27ae60; font-weight: bold;">{training_summary.get('final_train_loss', 'N/A')}</td>
              </tr>
              <tr style="background-color: #ecf0f1;">
                <td style="padding: 12px; border: 1px solid #bdc3c7; font-weight: bold;">Best Val Loss</td>
                <td style="padding: 12px; border: 1px solid #bdc3c7; color: #27ae60; font-weight: bold;">{training_summary.get('best_val_loss', 'N/A')}</td>
              </tr>
              <tr>
                <td style="padding: 12px; border: 1px solid #bdc3c7; font-weight: bold;">Training Duration</td>
                <td style="padding: 12px; border: 1px solid #bdc3c7;">{training_summary.get('duration', 'N/A')}</td>
              </tr>
              <tr style="background-color: #ecf0f1;">
                <td style="padding: 12px; border: 1px solid #bdc3c7; font-weight: bold;">Checkpoint Path</td>
                <td style="padding: 12px; border: 1px solid #bdc3c7; font-size: 12px;">{training_summary.get('checkpoint_path', 'N/A')}</td>
              </tr>
            </table>
          </div>
          
          <div style="margin: 20px 0; padding: 15px; background-color: #e8f5e9; border-left: 4px solid #4caf50; border-radius: 4px;">
            <p style="margin: 0; color: #2e7d32;">
              <strong>✅ Status:</strong> Training completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>
          </div>
          
          <div style="margin: 20px 0;">
            <h3 style="color: #34495e;">Additional Notes</h3>
            <p style="color: #555; line-height: 1.6;">
              {training_summary.get('notes', 'No additional notes.')}
            </p>
          </div>
          
          <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 12px;">
            <p>This is an automated notification from your training script.</p>
            <p>Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
          </div>
        </div>
      </body>
    </html>
    """
    
    # Attach HTML content
    html_part = MIMEText(html_content, 'html')
    msg.attach(html_part)
    
    try:
        # Connect to SMTP server
        print(f"Connecting to SMTP server: {smtp_server}:{smtp_port}")
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        
        # Login
        if sender_password:
            print(f"Logging in as: {sender_email}")
            server.login(sender_email, sender_password)
        
        # Send email
        print(f"Sending email to: {recipient_email}")
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        
        print(f"✅ Email sent successfully to {recipient_email}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        print("\nFor Gmail, you need to:")
        print("1. Enable 2-factor authentication on your Google account")
        print("2. Generate an App Password at: https://myaccount.google.com/apppasswords")
        print("3. Use the App Password instead of your regular password")
        return False


def get_training_summary_from_checkpoint(checkpoint_path: str) -> dict:
    """Extract training summary from checkpoint file"""
    import torch
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        summary = {
            'model_name': 'Snake Simple Distance Map Regression',
            'total_epochs': checkpoint.get('epoch', 'N/A'),
            'start_epoch': 101,
            'end_epoch': checkpoint.get('epoch', 'N/A'),
            'loss_type': 'Snake Simple' if checkpoint.get('snake_loss_enabled') else 'MSE',
            'final_train_loss': f"{checkpoint.get('train_loss', 0):.4f}" if checkpoint.get('train_loss') else 'N/A',
            'best_val_loss': f"{checkpoint.get('val_loss', 0):.4f}" if checkpoint.get('val_loss') else 'N/A',
            'checkpoint_path': checkpoint_path,
            'duration': 'See logs',
            'notes': f"Model trained with {checkpoint.get('snake_loss_type', 'MSE')} loss."
        }
        
        return summary
        
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        return {
            'model_name': 'Snake Simple Distance Map Regression',
            'total_epochs': 'Unknown',
            'start_epoch': 101,
            'end_epoch': 300,
            'loss_type': 'Snake Simple',
            'final_train_loss': 'See logs',
            'best_val_loss': 'See logs',
            'checkpoint_path': checkpoint_path,
            'duration': 'See logs',
            'notes': 'Training completed. Check logs for details.'
        }


if __name__ == "__main__":
    """
    Example usage:
    
    1. Set your email and password (use environment variables for security):
       export EMAIL_ADDRESS="your.email@gmail.com"
       export EMAIL_PASSWORD="your_app_password"
    
    2. Run this script:
       python send_training_email.py
    """
    
    import sys
    
    # Get email credentials from environment variables (recommended) or hardcode
    recipient_email = os.environ.get('EMAIL_ADDRESS', 'your.email@gmail.com')
    sender_password = os.environ.get('EMAIL_PASSWORD', None)
    
    # Find the latest best model checkpoint
    checkpoint_path = '/content/drive/MyDrive/ribbs/october/best_model.pth'
    
    # Get training summary
    if os.path.exists(checkpoint_path):
        summary = get_training_summary_from_checkpoint(checkpoint_path)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        summary = {
            'model_name': 'Snake Simple Distance Map Regression',
            'total_epochs': 300,
            'start_epoch': 101,
            'end_epoch': 300,
            'loss_type': 'Snake Simple',
            'final_train_loss': 'See logs',
            'best_val_loss': 'See logs',
            'checkpoint_path': 'N/A',
            'duration': 'See logs',
            'notes': 'Training completed. Check logs for details.'
        }
    
    # Send email
    if recipient_email == 'your.email@gmail.com':
        print("❌ Please set your email address!")
        print("\nOptions:")
        print("1. Set environment variable: export EMAIL_ADDRESS='your.email@gmail.com'")
        print("2. Or edit this script and replace 'your.email@gmail.com' with your actual email")
        sys.exit(1)
    
    if sender_password is None:
        print("❌ Please set your Gmail App Password!")
        print("\nSteps:")
        print("1. Go to: https://myaccount.google.com/apppasswords")
        print("2. Generate an App Password")
        print("3. Set environment variable: export EMAIL_PASSWORD='your_app_password'")
        sys.exit(1)
    
    success = send_training_complete_email(
        recipient_email=recipient_email,
        training_summary=summary,
        sender_password=sender_password
    )
    
    if success:
        print("\n✅ Notification sent successfully!")
    else:
        print("\n❌ Failed to send notification. Check the error message above.")

