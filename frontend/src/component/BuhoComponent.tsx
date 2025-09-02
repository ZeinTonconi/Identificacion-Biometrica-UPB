import Buho from '../assets/buho.png';

const BuhoComponent = () => {
  return (
    <div className="w-full md:w-1/2 flex items-center justify-center p-8">
      <img
        src={Buho}
        alt="Búho feliz"
        className="w-32 sm:w-40 md:w-96"
      />
    </div>
  )
}

export default BuhoComponent;
