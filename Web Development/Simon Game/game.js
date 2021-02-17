var buttonColours = ["red", "blue", "green", "yellow"];
var gamePattern = [];
var userClickedPattern = [];
var gameStart = false;
var level = 0;
var currentLvl = 0;

$(document).keydown(function(event) {
  if (!gameStart) {
    nextSequence();
    gameStart = true;
  }
});

$(".btn").click(function() {
  var userChosenColour = this.id;
  userClickedPattern.push(userChosenColour);
  $("#" + userChosenColour).fadeOut(100).fadeIn(100);
  animatePress(userChosenColour);
  playSound(userChosenColour);

  if (checkIfWin(currentLvl))
  {
    if(currentLvl === gamePattern.length - 1)
    {
      setTimeout(function() {
        nextSequence();
        userClickedPattern = [];
      }, 1000);
      currentLvl = 0;
    }
    else
      currentLvl++;
  }
  else {
    var wrongSound = new Audio("./sounds/wrong.mp3");
    wrongSound.play();
    $("body").addClass("game-over");
    setTimeout(function() {
      $("body").removeClass("game-over")
    }, 200);
    $("#level-title").html("Game Over<br>Press Any Key to Restart");
    startOver();
  }
});

function nextSequence() {
  var randomNumber = Math.floor(Math.random() * 4);
  var randomChosenColour = buttonColours[randomNumber];
  gamePattern.push(randomChosenColour);
  $("#" + randomChosenColour).fadeOut(100).fadeIn(100);
  animatePress(randomChosenColour);
  playSound(randomChosenColour);
  level++;
  $("#level-title").text("Level " + level);
}



function playSound(chosenColour) {
  var audio = new Audio("./sounds/" + chosenColour + ".mp3");
  audio.play();
}

function animatePress(currentcolour) {
  $("#" + currentcolour).addClass("pressed");
  setTimeout(function() {
    $("#" + currentcolour).removeClass("pressed");
  }, 100);
}

function checkIfWin(lvl) {
  if (gamePattern[lvl] === userClickedPattern[lvl])
    return true;
  else
    return false;
}

function startOver()
{
  gameStart = false;
  level = 0;
  gamePattern = [];
}
