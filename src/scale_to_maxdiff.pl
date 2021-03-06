#!/usr/bin/perl
#
#
#   scale_to_maxdiff.pl
#
#   - given a set of MaxDiff questions and a set of word pairs rated on a scale,
#     answer the MaxDiff questions
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
#   check command line arguments
#
if ($#ARGV != 2) {
  print "\n\nUsage:\n\n";
  print "scale_to_maxdiff.pl <input file of MaxDiff questions> <input file of scaled pairs> <output file of answers to MaxDiff questions>\n\n";
  exit;
}
#
#   input file of MaxDiff questions
#
$max_file = $ARGV[0];
#
#   input file of scaled pairs
#
$sca_file = $ARGV[1];
#
#   output file of answers to MaxDiff questions
#
$out_file = $ARGV[2];
#
#
#
#
#
#
#   read in the scaled pairs
#
print "reading file of scaled pairs $sca_file ...\n";
#
%pair2scale = ();
$num_pairs  = 0;
#
open(INF, "< $sca_file");
#
while ($line = <INF>) {
  #
  #   typical $line:
  #
  #    52.0 "album:songs"
  #    48.0 "army:soldier"
  #    44.0 "book:page"
  #
  if ($line =~ /^\#/) { next; }        # skip comments
  if ($line =~ /(\S+)\s+(\S+)/) {
    $scale = $1;
    $pair  = $2;
    $pair2scale{$pair} = $scale;
    $num_pairs++;
  }
}
#
close(INF);
#
print "... read $num_pairs pairs ...\n";
print "... done.\n";
#
#
#
#
#
#   read in the MaxDiff questions
#
print "reading file of MaxDiff questions $max_file ...\n";
#
@questions = ();
$num_quest = 0;
#
open(INF, "< $max_file");
#
while ($line = <INF>) {
  #
  #   typical $line:
  #
  #   "school:fish","library:book","flock:sheep","flock:bird"
  #   "class:student","pride:lion","band:musician","geese:gaggle"
  #
  if ($line =~ /^\#/) { next; }        # skip comments
  if ($line =~ /(\".+\")/) {
    $question = $1;
    push(@questions, $question);
    $num_quest++;
  }
}
#
close(INF);
#
print "... read $num_quest questions ...\n";
print "... done.\n";
#
#
#
#
#
#
#
#   answer the questions
#
print "writing answers to $out_file ...\n";
#
open(OUTF, "> $out_file");
#
print OUTF "#\n";
print OUTF "#   Generated by:                 scale_to_maxdiff.pl\n";
print OUTF "#   Scaled Pairs File:            $sca_file\n";
print OUTF "#   MaxDiff Questions File:       $max_file\n";
print OUTF "#   MaxDiff Answers File:         $out_file\n";
print OUTF "#   Number of Unique Pairs:       $num_pairs\n";
print OUTF "#   Number of MaxDiff Questions:  $num_quest\n";
print OUTF "#\n";
print OUTF "#   relation1 relation2 relation3 relation4 least_illustrative most_illustrative\n";
print OUTF "#\n";
#
foreach $question (@questions) {
  @pairs = split(/\,/, $question);
  for ($i = 0; $i < 4; $i++) {
    $pair  = $pairs[$i];
    $scale = $pair2scale{$pair};
    if (! defined($scale)) {
      die "ERROR: $pair is in $max_file but not in $sca_file\n";
    }
    if (($i == 0) || ($scale > $max_scale)) {
      $max_scale = $scale;
      $max_pair  = $pair;
    }
    if (($i == 0) || ($scale < $min_scale)) {
      $min_scale = $scale;
      $min_pair  = $pair;
    }
  }
  push(@pairs, $min_pair);
  push(@pairs, $max_pair);
  $out_string = join(" ", @pairs);
  print OUTF "$out_string\n";
}
#
close(OUTF);
#
print "... done.\n";
#
#
#
#
#

