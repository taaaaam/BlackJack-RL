from classes import Environment
from Q_learning import PlayerQL
from MC_learning import PlayerMC
from baseline import PlayerBASE
from TD_learning import PlayerTD
from SARSA_learning import PlayerSARSA


if __name__ == "__main__":
      player = PlayerQL(epsilon = .2, alpha = .1)
      env = Environment(player)
      env.q_train(500000)
      result = env.test(num_epochs = 20000)
      print('Q_LEARNING: eps = %s, step-size = %s, WIN AVERAGE: %s' 
            % (player.epsilon, player.alpha, result))

      player = PlayerMC(epsilon = 0.2)
      env = Environment(player)
      env.mc_train(500000)
      result = env.test(20000)
      print('FIRST MC: eps = %s, WIN AVERAGE: %s' % (env.player.epsilon, result))

      player = PlayerBASE()
      env = Environment(player)
      result = env.test(20000)
      print('BASE: WIN AVERAGE: %s' % (result))
      
      player = PlayerTD(alpha = .1, epsilon=.2)
      env = Environment(player)
      env.td_train(500000)
      result = env.test(20000)
      print('TD: eps = %s, step-size = %s, WIN AVERAGE: %s' % (env.player.epsilon, env.player.alpha, result))
      
      player = PlayerSARSA(alpha = .05, epsilon=.4)
      env = Environment(player)
      env.sarsa_train(500000)
      result = env.test(20000)
      print('SARSA: eps = %s, step-size = %s, WIN AVERAGE: %s' % (env.player.epsilon, env.player.alpha, result))

      

    
      


    
    
