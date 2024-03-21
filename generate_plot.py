import pandas as pd
import matplotlib.pyplot as plt

# Load losses from CSV
losses_df = pd.read_csv('gan_losses.csv')

# Plot
plt.figure(figsize=(10, 5))
plt.plot(losses_df['Generator Loss'], label='Generator Loss')
plt.plot(losses_df['Discriminator Loss'], label='Discriminator Loss')
plt.title('GAN Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

print("saving plot...")
# Save plot to file
plt.savefig('gan_losses.png')
plt.close()
