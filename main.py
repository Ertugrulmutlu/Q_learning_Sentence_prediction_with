import numpy as np
import matplotlib.pyplot as plt

# Sentence to be generated
sentence = "Hello, how are you?"
# List of characters without spaces and punctuation marks
characters = " abcdefghijklmnopqrstuvwxyz,?"

# RL parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor for future rewards
num_episodes = 10  # Number of training episodesdddd

# Initialize Q values to zerod
Q = np.zeros((len(sentence), len(characters)))

# Training
for episode in range(num_episodes):
    state = 0  # Initial state
    done = False
    while not done:
        # Select a random action
        action = np.random.randint(len(characters))
        print(action)
        # If reached the last state, exit the loop
        if state == len(sentence) - 1:
            done = True
        else:
            # Determine the next state
            next_state = state + 1
            
            # If the selected character matches the character in the target sentence at the same position, give a reward
            if sentence[state] == characters[action]:
                reward = 1
            else:
                reward = 0
            
            # Update Q values
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            
            # Update the next state
            state = next_state

# Generate the predicted sentence
predicted_sentence = ""
state = 0
while state < len(sentence):
    # Select the action with the highest Q value
    action = np.argmax(Q[state])
    # Add the character to the predicted sentence
    predicted_sentence += characters[action]
    # Update the next state
    state += 1

# Print the predicted sentence
print("Predicted Sentence:", predicted_sentence)

# Visualization of Q values
plt.figure(figsize=(12, 8))
plt.imshow(Q, cmap='hot', interpolation='nearest')
plt.colorbar(label='Q Value')
plt.title("Distribution of Q Values")
plt.xlabel("Action (Character)")
plt.ylabel("State")
plt.xticks(np.arange(len(characters)), list(characters))
plt.yticks(np.arange(len(sentence)), list(sentence))
plt.show()