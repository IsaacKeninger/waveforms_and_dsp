let attackValue;
let releaseValue;

function init() {
  getAttack()
  getRelease()
  keyDown()
}

function getAttack(){
    document.getElementById("attack").addEventListener("input", function() {
    attackValue = this.value;
    document.getElementById("Aoutput").innerText = "Attack: " + attackValue;

    if (attackValue == ''){
            document.getElementById("Aoutput").innerText = "Attack: " + attackValue;
    }
  })
}

function getRelease(){
  document.getElementById("release").addEventListener("input", function() {
  releaseValue = this.value;
  document.getElementById("Routput").innerText = "Release: " + releaseValue;
  });
}

function keyDown(){
    document.addEventListener('keydown',(event)=>{
        if (event.key == 'm'){
            document.getElementById("Koutput").innerText = 'Keydown event: ' + event.key;
        } else{
            document.getElementById("Koutput").innerText = ''
        }
    })
}
window.onload = init;