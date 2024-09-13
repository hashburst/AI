def calculate_reward(rejected_shares, accepted_shares, accepted_user, total_accepted_cluster, T_gpu):
                          # Soglie e parametri
                          T_max = 90  # Temperatura massima tollerata in gradi Celsius
                          max_reward = 100  # Reward base
                          penalty_temp = max(0, T_gpu - T_max)  # Penalizzazione per temperatura
                      
                          # Penalizzazione se non si raggiunge il 100% di accepted shares
                          penalty_accepted = 100 - accepted_shares if accepted_shares < 100 else 0
                      
                          # Contributo proporzionale dell'utente nel cluster
                          if total_accepted_cluster > 0:
                              share_contribution = 1 - (accepted_user / total_accepted_cluster)
                          else:
                              share_contribution = 1  # Penalizzazione massima se non ci sono accepted shares nel cluster
                      
                          # Calcolo della reward finale
                          reward = max_reward - rejected_shares - penalty_temp - (penalty_accepted * share_contribution)
                      
                          # La reward non può essere negativa
                          return max(0, reward)
