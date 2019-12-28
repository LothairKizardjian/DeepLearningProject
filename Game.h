
const int MaxGames = 500000;
const int MaxMoveGame = 1000;
int nbGames = 0;
int nbMovesSGFGame [MaxGames];
//Move proGame [MaxGames] [MaxMoveGame];
Move *proGame [MaxGames];
char winner [MaxGames];

bool proGameAllocated = false;

class PositionSGF {
 public:
  int game;
  short int move;
};

int nbPositionsSGF = 0;
PositionSGF * positionSGF = NULL;
int startShuffle = 0;

void shuffle () {
  fprintf (stderr, "nbPositionsSGF = %d\n", nbPositionsSGF);
  for (int i = startShuffle; i < nbPositionsSGF; i++) {
    int other = startShuffle + rand () % (nbPositionsSGF - startShuffle);
    PositionSGF tmp = positionSGF [i];
    positionSGF [i] = positionSGF [other];
    positionSGF [other] = tmp;
  }
}

void loadGames (char * name) {
  if (!proGameAllocated) {
    proGameAllocated = true;
    for (int i = 0; i < MaxGames; i++)
      proGame [i] = new Move [MaxMoveGame];
  }

  FILE * fp = fopen (name, "r");
  nbPositionsSGF = 0;
  nbGames = 0;
  if (positionSGF == NULL)
    positionSGF = new PositionSGF [MaxGames * MaxMoveGame];
  if (fp != NULL) {
    char game [1000];
    int res = 0;
    while ((res != -1) && (nbGames < MaxGames)) {
      res = fscanf (fp, "%s", game);
      if (res != -1) {
        FILE * sgf = fopen (game, "r");
        if (sgf != NULL) {
          Board b = board;
	  b.loadSGF (sgf);
	  //if (nbGames == 180)
	  //b.print (stderr);
	  fprintf (stderr, "%s ", game);
	  winner [nbGames] = b.winner;
	  nbMovesSGFGame [nbGames] = 0;
	  for (int i = 0; i < b.NbMovesPlayed; i++) {
	    //if (b.Moves [i] != b.passe) {
	    Move m;
	    m.inter = moveInter [b.Moves [i]];
	    m.color = b.colorMove [i];
	    if (i == 0)
	      m.val = 0.0;
	    else
	      m.val = b.val [i - 1];
	    //fprintf (stderr, "m.val = %2.3f ", m.val);
	    if (nbMovesSGFGame [nbGames] < MaxMoveGame - 1) {
	      proGame [nbGames] [nbMovesSGFGame [nbGames]] = m;
	      positionSGF [nbPositionsSGF].game = nbGames;
	      positionSGF [nbPositionsSGF].move = nbMovesSGFGame [nbGames];
	      nbPositionsSGF++;
	      nbMovesSGFGame [nbGames]++;
	    }
	    //}
	  }
	  nbGames++;
          fclose (sgf);
        }
      }
    }
    fclose (fp);
  }
  shuffle ();
}

void writeGamesData (char * name) {
  FILE * fp = fopen (name, "w");
  if (fp != NULL) {
    fprintf (fp, "%d\n", nbGames);
    for (int g = 0; g < nbGames; g++) {
      fprintf (fp, "%c ", winner [g]);
      fprintf (fp, "%d ", nbMovesSGFGame [g]);
      for (int i = 0; i < nbMovesSGFGame [g]; i++) {
	fprintf (fp, "%d %d ", proGame [g] [i].inter, proGame [g] [i].color);
      }
      fprintf (fp, "\n");
    }
    fclose (fp);
  }
}

void loadGamesData (char * name) {
  if (!proGameAllocated) {
    proGameAllocated = true;
    for (int i = 0; i < MaxGames; i++)
      proGame [i] = new Move [MaxMoveGame];
  }

  FILE * fp = fopen (name, "r");
  if (fp == NULL) {
    fprintf (stderr, "Error loading %s\n", name);
    return;
  }
  char s [1000];
  nbPositionsSGF = 0;
  if (positionSGF == NULL)
    positionSGF = new PositionSGF [MaxGames * MaxMoveGame];
  if (fp != NULL) {
    fscanf (fp, "%d", &nbGames);
    for (int g = 0; g < nbGames; g++) {
      fscanf (fp, "%s ", s);
      winner [g] = s [0];
      fscanf (fp, "%d", &nbMovesSGFGame [g]);
      Board b = board;
      for (int i = 0; i < nbMovesSGFGame [g]; i++) {
	fscanf (fp, "%d %d", &proGame [g] [i].inter, &proGame [g] [i].color);
	positionSGF [nbPositionsSGF].game = g;
	positionSGF [nbPositionsSGF].move = i;
	int move = proGame [g] [i].inter;
	if (move != 361)
	  if ((b.board [interMove [move]] != Empty) || !b.legalMove (interMove [move], b.turn)) {
	    b.print (stderr);
	    fprintf (stderr, "b.length = %d\n", b.length);
	    fprintf (stderr, "g = %d, nbPositionsSGF = %d, nbMoves [%d] = %d, winner [%d] = %c\n",
		     g, nbPositionsSGF, g, nbMovesSGFGame [g], g, winner [g]);
	    fprintf (stderr, "Bug play, move = %d (%d,%d)\n", move, move / 19, move % 19);
	    exit (1);
	  }
	b.play (move);
	nbPositionsSGF++;
      }
    }
    //fprintf (stderr, "nbPositionsSGF = %d\n", nbPositionsSGF);
    shuffle ();
    fclose (fp);
  }
}

void writeGamesDataVal (char * name) {
  FILE * fp = fopen (name, "w");
  if (fp != NULL) {
    fprintf (fp, "%d\n", nbGames);
    for (int g = 0; g < nbGames; g++) {
      fprintf (fp, "%c ", winner [g]);
      fprintf (fp, "%d ", nbMovesSGFGame [g]);
      for (int i = 0; i < nbMovesSGFGame [g]; i++) {
	fprintf (fp, "%d %2.3f %d ", proGame [g] [i].inter, proGame [g] [i].val, proGame [g] [i].color);
      }
      fprintf (fp, "\n");
    }
    fclose (fp);
  }
}

void loadGamesDataVal (char * name) {
  FILE * fp = fopen (name, "r");
  if (fp == NULL) {
    fprintf (stderr, "Error loading %s\n", name);
    return;
  }
  char s [1000];
  nbPositionsSGF = 0;
  if (positionSGF == NULL)
    positionSGF = new PositionSGF [MaxGames * MaxMoveGame];
  if (fp != NULL) {
    fscanf (fp, "%d", &nbGames);
    for (int g = 0; g < nbGames; g++) {
      fscanf (fp, "%s ", s);
      winner [g] = s [0];
      fscanf (fp, "%d", &nbMovesSGFGame [g]);
      Board b = board;
      for (int i = 0; i < nbMovesSGFGame [g]; i++) {
	fscanf (fp, "%d %f %d ", &proGame [g] [i].inter, &proGame [g] [i].val, &proGame [g] [i].color);
	positionSGF [nbPositionsSGF].game = g;
	positionSGF [nbPositionsSGF].move = i;
	int move = proGame [g] [i].inter;
	if (move != 361)
	  if ((b.board [interMove [move]] != Empty) || !b.legalMove (interMove [move], b.turn)) {
	    b.print (stderr);
	    fprintf (stderr, "b.length = %d\n", b.length);
	    fprintf (stderr, "g = %d, nbPositionsSGF = %d, nbMoves [%d] = %d, winner [%d] = %c\n",
		     g, nbPositionsSGF, g, nbMovesSGFGame [g], g, winner [g]);
	    fprintf (stderr, "Bug play, move = %d (%d,%d)\n", move, move / 19, move % 19);
	    exit (1);
	  }
	b.play (move);
	nbPositionsSGF++;
      }
    }
    //fprintf (stderr, "nbPositionsSGF = %d\n", nbPositionsSGF);
    shuffle ();
    fclose (fp);
  }
}

const int Planes = 3;

char historyBoard [MaxPlayoutLength] [MaxSize];

void play (Board * board, int move) {
  if (move != 361)
    if (board->board [interMove [move]] != Empty) {
      board->print (stderr);
      fprintf (stderr, "board->length = %d\n", board->length);
      fprintf (stderr, "Bug play, move = %d (%d,%d)\n", move, move / 19, move % 19);
      exit (1);
    }
  board->play (move);
  memcpy (historyBoard [board->length], board->board, MaxSize);
  //fprintf (stderr, "move = %d (%d,%d)\n", move, move / 19, move % 19);
  //board->print (stderr);
}

float input [19] [19] [2 * Planes + 3];

void encode (Board * board) {
  if (board->turn == Black) {
    for (int i = 0; i < 19; i++)
      for (int j = 0; j < 19; j++)
	input [i] [j] [0] = 1.0;
  }
  else {
    for (int i = 0; i < 19; i++)
      for (int j = 0; j < 19; j++)
	input [i] [j] [0] = 0.0;
  }
  board->computeLostLadders ();
  for (int i = 0; i < 19; i++)
    for (int j = 0; j < 19; j++) 
      input [i] [j] [1] = board->isLostLadder [interMove [19 * i + j]];
  int start = 2;
  int other = Black;
  if (board->turn == Black)
    other = White;
  int current = board->length;
  for (int plane = 0; plane < Planes; plane++) {
    if (current < 0) {
      for (int i = 0; i < 19; i++)
	for (int j = 0; j < 19; j++) {
	  input [i] [j] [start + 2 * plane] = 0.0;
	  input [i] [j] [start + 2 * plane + 1] = 0.0;
	}
    }
    else {
      for (int i = 0; i < 19; i++)
	for (int j = 0; j < 19; j++) {
	  if (historyBoard [current] [interMove [19 * i + j]] == board->turn) {
	    input [i] [j] [start + 2 * plane] = 1.0;
	    input [i] [j] [start + 2 * plane + 1] = 0.0;
	  }
	  else if (historyBoard [current] [interMove [19 * i + j]] == other) {
	    input [i] [j] [start + 2 * plane] = 0.0;
	    input [i] [j] [start + 2 * plane + 1] = 1.0;
	  }
	  else {
	    input [i] [j] [start + 2 * plane] = 0.0;
	    input [i] [j] [start + 2 * plane + 1] = 0.0;
	  }
	}
    }
    current--;
  }   
}





