#!/usr/bin/perl
#
#
#   score_maxdiff.pl
#
#   - evaluate a set of answers to MaxDiff questions by comparing them with
#     the majority vote of Mechanical Turkers
#
#
#
#
#   Peter Turney
#   December 19, 2011
#
#
#
#
#
#
#
#
#   check command line arguments
#
if ($#ARGV != 2) {
  print "\n\nUsage:\n\n";
  print "score_maxdiff.pl <input file of Mechanical Turk answers to MaxDiff questions> <input file of MaxDiff answers to be evaluated> <output file of results>\n\n";
  exit;
}
#
#
#
#
#
#   input file of Mechanical Turk answers to MaxDiff questions
#
$turk_file = $ARGV[0];
#
#   input file of MaxDiff answers to be evaluated
#
$test_file = $ARGV[1];
#
#   output file of results
#
$out_file = $ARGV[2];
#
#
#
#
#
#
#   read in the Mechanical Turk file
#
print "reading the Mechanical Turk answers to MaxDiff questions $turk_file ...\n";
#
@questions      = ();    # list of unique MaxDiff questions
%question2qnum  = ();    # mapping of question string to question number
%pairqnum2least = ();    # mapping of word pair and question number to number of Turkers who chose pair as Least Illustrative
%pairqnum2most  = ();    # mapping of word pair and question number to number of Turkers who chose pair as Most Illustrative
$num_quest      = 0;     # number of unique MaxDiff questions
#
open(INF, "< $turk_file");
#
while ($line = <INF>) {
  if ($line =~ /^\#/) { next; }                        # skip comments
  @fields = split(/\s+/, $line);
  $question = join(" ", ($fields[0], $fields[1], $fields[2], $fields[3]));
  if (! defined($question2qnum{$question})) {          # add a new question to the list
    $question2qnum{$question} = $num_quest;
    push(@questions, $question);
    for ($i = 0; $i < 4; $i++) {
      $pair                      = $fields[$i];
      $pairqnum                  = "$pair $num_quest";
      $pairqnum2least{$pairqnum} = 0;
      $pairqnum2most{$pairqnum}  = 0;
    }
    $num_quest++;
  }
  $qnum     = $question2qnum{$question};
  $least    = $fields[4];
  $pairqnum = "$least $qnum";
  $pairqnum2least{$pairqnum}++;
  $most     = $fields[5];
  $pairqnum = "$most $qnum";
  $pairqnum2most{$pairqnum}++;
}
#
close(INF);
#
print "... read $num_quest unique MaxDiff questions ...\n";
print "... done.\n";
#
#
#
#
#   read the MaxDiff answers to be evaluated
#
print "reading the MaxDiff answers to be evaluated $test_file ...\n";
#
@most_list  = ();    # list of choices for Most Illustrative
@least_list = ();    # list of choices for Least Illustrative
$num_test   = 0;     # number of MaxDiff questions
#
open(INF, "< $test_file");
#
while ($line = <INF>) {
  if ($line =~ /^\#/) { next; }                        # skip comments
  @fields = split(/\s+/, $line);
  $question = join(" ", ($fields[0], $fields[1], $fields[2], $fields[3]));
  $qnum     = $question2qnum{$question};
  if (! defined($qnum)) {
    die "ERROR: the following question is not in the Turk file: $question\n";
  }
  $least = $fields[4];
  $most  = $fields[5];
  $most_list[$qnum]  = $most;
  $least_list[$qnum] = $least;
  $num_test++;
}
#
close(INF);
#
print "... read $num_test answers ...\n";
print "... done.\n";
#
if ($num_quest != $num_test) {
  die "ERROR: found $num_quest questions in $turk_file but $num_test answers in $test_file\n";
}
#
#
#
#
#
#   score the answers
#
print "scoring the answers and writing a summary to $out_file ...\n";
#
$num_least_right = 0;
$num_least_wrong = 0;
$num_most_right  = 0;
$num_most_wrong  = 0;
#
for ($qnum = 0; $qnum < $num_quest; $qnum++) {
  $guess_least       = $least_list[$qnum];
  $guess_most        = $most_list[$qnum];
  if ((! defined($guess_least)) || (! defined($guess_most))) { 
    die "ERROR: question number $qnum not answered\n";
  }
  $votes_guess_least = $pairqnum2least{"$guess_least $qnum"};
  $votes_guess_most  = $pairqnum2most{"$guess_most $qnum"};
  $max_votes_least   = 0;
  $max_votes_most    = 0;
  $question          = $questions[$qnum];
  @fields            = split(/\s+/, $question);
  for ($i = 0; $i < 4; $i++) {
    $pair            = $fields[$i];
    $pairqnum        = "$pair $qnum";
    $num_votes_least = $pairqnum2least{$pairqnum};
    $num_votes_most  = $pairqnum2most{$pairqnum};
    if ($num_votes_least > $max_votes_least) {
      $max_votes_least = $num_votes_least;
    }
    if ($num_votes_most > $max_votes_most) {
      $max_votes_most = $num_votes_most;
    }
  }
  if ($votes_guess_least == $max_votes_least) {
    $num_least_right++;
  }
  else {
    $num_least_wrong++;
  }
  if ($votes_guess_most == $max_votes_most) {
    $num_most_right++;
  }
  else {
    $num_most_wrong++;
  }
}
#
$acc_most  = sprintf("%5.1f", (100 * $num_most_right / $num_quest));
$acc_least = sprintf("%5.1f", (100 * $num_least_right / $num_quest));
$acc_all   = sprintf("%5.1f", (100 * ($num_most_right + $num_least_right) / (2 * $num_quest)));
#
open(OUTF, "> $out_file");
#
print OUTF "\n";
print OUTF "Generated by:                                     score_maxdiff.pl\n";
print OUTF "Mechanical Turk File:                             $turk_file\n";
print OUTF "Test File:                                        $test_file\n";
print OUTF "Number of MaxDiff Questions:                      $num_quest\n";
print OUTF "Number of Least Illustrative Guessed Correctly:   $num_least_right\n";
print OUTF "Number of Least Illustrative Guessed Incorrectly: $num_least_wrong\n";
print OUTF "Accuracy of Least Illustrative Guesses:           $acc_least\%\n";
print OUTF "Number of Most Illustrative Guessed Correctly:    $num_most_right\n";
print OUTF "Number of Most Illustrative Guessed Incorrectly:  $num_most_wrong\n";
print OUTF "Accuracy of Most Illustrative Guesses:            $acc_most\%\n";
print OUTF "Overall Accuracy:                                 $acc_all\%\n";
#
close(OUTF);
#
print "... done.\n";
#
#
#
#