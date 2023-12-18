BlackJack:
	echo "#!/bin/bash" > BlackJack
	echo "python3 BlackJack.py \"\$$@\"" >> BlackJack
	chmod u+x BlackJack
	cat INSTRUCTIONS.txt